import os
import re
from typing import List, Dict, Tuple, Union

from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate

class RAGModel:
    def __init__(self, kb_path: str, google_api_key: str):
        self.kb_path = kb_path
        self.google_api_key = google_api_key

        # Configure embeddings & LLM
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=self.google_api_key,
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",  # fast & cost-effective
            temperature=0.3,
            google_api_key=self.google_api_key,
        )
        self.vector_db = None
        self.retriever = None

        # Build FAISS from KB
        self._build_vector_db()

    def _build_vector_db(self):
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"KB file not found: {self.kb_path}")

        # Load .docx and split via RecursiveCharacterTextSplitter
        docs = Docx2txtLoader(self.kb_path).load()  # one big Document
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800,
            chunk_overlap=300,
            separators=["\n## ", "\n### ", "\n", ". ", " ", ""],  # coarse -> fine
        )

        texts = []
        for d in docs:
            texts.extend(splitter.split_text(d.page_content))

        # Build FAISS index and create retriever
        self.vector_db = FAISS.from_texts(texts, embedding=self.embeddings)
        self.vector_db.save_local("index_db")
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 6})

    def _retrieve(self, query: str) -> List[str]:
        docs = self.retriever.invoke(query)
        return [d.page_content for d in docs]

    @staticmethod
    def _build_prompt(user_req: str, kb_context: List[str], include_neg: bool, refinements: str) -> PromptTemplate:
        context_text = "\n\n".join(kb_context)
        neg = "- Include negative/edge scenarios and failure handling.\n" if include_neg else "- Focus on happy-path; negative scenarios optional.\n"
        ref = f"\nAdditional refinements:\n{refinements}\n" if refinements.strip() else ""
        template = f"""
You are an AI agent that generates **structured Agile user stories** with **acceptance criteria**.
Leverage the retrieved **knowledge base** (templates, compliance rules, historical examples, numbered requirements).
Respond in **Markdown**.

### Requirement (from user)
{{user_req}}

### Retrieved KB Context (RAG)
{context_text}

### Instructions
- Use format: *As a <role>, I want <capability>, so that <benefit>*.
- 2–3 stories per feature; each \u2264 120 words.
- **Acceptance Criteria** in **Gherkin** (Given/When/Then), 3–5 bullets per story.
- Integrate applicable **Business Rules** and **Compliance** (security baseline, GDPR/CCPA, WCAG 2.1 AA, PCI DSS, audit/retention, etc.).
{neg}- Keep output concise, precise, implementation-agnostic.
{ref}

### Output
1) **User Stories**
2) **Acceptance Criteria** (per story)
3) **Non-Functional Criteria**
4) **Standards Validation Summary** (gaps)
"""
        return PromptTemplate(template=template, input_variables=["user_req"])

    def generate(
        self,
        user_req: str,
        include_neg: bool,
        refinements: str = "",
        extra_query_terms: List[str] = None,
    ) -> str:
        # Augment retrieval when corrective RAG finds gaps
        augmented_query = user_req
        if extra_query_terms:
            augmented_query = f"{user_req}\n" + " ".join(extra_query_terms)

        kb_context = self._retrieve(augmented_query)
        prompt = self._build_prompt(user_req, kb_context, include_neg, refinements)
        chain = prompt | self.llm
        result = chain.invoke({"user_req": user_req})

        # Ensure the output content is a single string
        if isinstance(result.content, list):
            # If it's a list of contents (e.g., multimodal), join them into a single string
            processed_content = " ".join([str(item) for item in result.content])
        else:
            processed_content = str(result.content) # Ensure it's a string

        return processed_content

    # -----------------------
    # Corrective RAG helpers
    # -----------------------
    def _validate_output(self, output: str, include_neg: bool) -> Tuple[bool, List[str]]:
        """
        Returns (passed, gaps). 'gaps' is a list of human-readable issues to fix.
        Keep checks simple & robust.
        """
        gaps = []

        # Required section headers
        required_sections = [
            "User Stories",
            "Acceptance Criteria",
            "Non-Functional Criteria",
            "Standards Validation Summary",
        ]
        for sec in required_sections:
            if sec.lower() not in output.lower():
                gaps.append(f"Missing section: **{sec}**")

        # Story count & format
        story_lines = re.findall(r"(?:^|\n)\s*(?:\*?\s*)As a .*?, I want .*?, so that .*?(\.|$) selections=false", output, flags=re.IGNORECASE)
        if not (2 <= len(story_lines) <= 3):
            gaps.append("Ensure **2–3 user stories** using the format: *As a <role>, I want <capability>, so that <benefit>*.")

        # Story length (<=120 words per story line)
        story_line_texts = re.findall(r"(?:^|\n)\s*(?:\*?\s*)(As a .*?, I want .*?, so that .*?)(?:\.|$) selections=false", output, flags=re.IGNORECASE)
        for s in story_line_texts:
            words = re.findall(r"\w+", s)
            if len(words) > 120:
                gaps.append("Each **user story** should be **\u2264 120 words**. Shorten long stories.")

        # Gherkin presence
        if not re.search(r"\bGiven\b", output, re.IGNORECASE) or not re.search(r"\bWhen\b", output, re.IGNORECASE) or not re.search(r"\bThen\b", output, re.IGNORECASE):
            gaps.append("Add **Gherkin-style Acceptance Criteria** with **Given/When/Then** (3–5 bullets per story).")

        # Compliance mentions
        compliance_terms = ["GDPR", "CCPA", "WCAG", "PCI", "security", "audit", "retention"]
        if not any(term.lower() in output.lower() for term in compliance_terms):
            gaps.append("Explicitly reference **GDPR**, **CCPA**, **WCAG 2.1 AA**, **PCI DSS**, **security baseline**, **audit & retention** in criteria or summary.")

        # Negative / edge scenarios if required
        if include_neg:
            neg_keywords = ["negative", "edge", "error", "failure", "unauthorized", "invalid", "timeout"]
            if not any(k in output.lower() for k in neg_keywords):
                gaps.append("Include **negative/edge scenarios** and **failure handling** in acceptance criteria.")

        passed = len(gaps) == 0
        return passed, gaps

    def _build_refinements_from_gaps(self, gaps: List[str]) -> Tuple[str, List[str]]:
        """
        Turn gaps into a concise refinement string for the LLM and extra query terms for retrieval.
        """
        # Refinements as a clear checklist to the model
        refinements = "Please correct the following:\n" + "\n".join([f"- {g}" for g in gaps])

        # Extra query terms to pull specific KB rules in next iteration
        extra_terms = []
        for g in gaps:
            gl = g.lower()
            if "gdpr" in gl or "ccpa" in gl:
                extra_terms += ["GDPR", "CCPA", "privacy", "data protection", "retention"]
            if "wcag" in gl:
                extra_terms += ["WCAG 2.1 AA", "accessibility", "screen reader", "contrast"]
            if "pci" in gl:
                extra_terms += ["PCI DSS", "payment", "cardholder", "encryption"]
            if "security" in gl:
                extra_terms += ["security baseline", "authentication", "authorization", "audit"]
            if "gherkin" in gl or "given/when/then" in gl:
                extra_terms += ["Given When Then", "acceptance criteria", "Gherkin"]
            if "negative" in gl or "edge" in gl or "failure" in gl:
                extra_terms += ["error handling", "edge cases", "invalid input", "unauthorized", "timeout"]

        # Deduplicate while preserving order
        seen = set()
        extra_terms_unique = []
        for t in extra_terms:
            if t not in seen:
                extra_terms_unique.append(t)
                seen.add(t)

        return refinements, extra_terms_unique

    def corrective_generate(
        self,
        user_req: str,
        include_neg: bool,
        initial_refinements: str = "",
        max_loops: int = 3,
    ) -> str:
        """
        Generate -> Validate -> Refine -> Regenerate with augmented retrieval.
        Stops when validation passes or max_loops reached.
        """
        refinements = initial_refinements or ""
        extra_terms: List[str] = []

        last_output = ""
        for i in range(1, max_loops + 1):
            # 1) Generate with current refinements & extra retrieval queries
            output = self.generate(
                user_req=user_req,
                include_neg=include_neg,
                refinements=refinements,
                extra_query_terms=extra_terms,
            )
            last_output = output

            # 2) Validate
            passed, gaps = self._validate_output(output, include_neg)
            print(f"\n[Validation Iteration {i}] Passed: {passed}")
            if gaps:
                for g in gaps:
                    print(f" - {g}")

            if passed:
                print("\n[Corrective RAG] Validation passed.")
                return output

            # 3) Build refinements & augment retrieval terms
            refine_msg, new_terms = self._build_refinements_from_gaps(gaps)
            # Accumulate refinements (keep simple, append)
            if refinements:
                refinements += "\n" + refine_msg
            else:
                refinements = refine_msg
            # Merge extra terms
            for t in new_terms:
                if t not in extra_terms:
                    extra_terms.append(t)

            print("\n[Corrective RAG] Regenerating with refinements & augmented retrieval...")

        print("\n[Corrective RAG] Max loops reached. Returning last output with potential gaps.")
        return last_output


def main():
    print("=== AI-Powered User Story & Acceptance Criteria Generator (RAG + Recursive Split + FAISS) ===")

   
   # Get Google API key from environment if already set; otherwise prompt user
    google_key = os.environ.get("GOOGLE_API_KEY")
    if not google_key:
        google_key = input("Enter your Google API Key: ").strip()
    if not google_key:
        raise ValueError("Google API key is required.")
    os.environ["GOOGLE_API_KEY"] = google_key
   
    # Prompt for KB path dynamically in VS Code
    kb_path = input("Enter KB .docx path (example: C:\\Users\\paila\\OneDrive\\Desktop\\digital_wallet_business_compliance_rules.docx): ").strip()
   
    # Remove accidental quotes if user pastes a path with quotes
    kb_path = kb_path.strip('"').strip("'")
   
    if not kb_path:
        print("No file path entered. Please provide a valid .docx path.")
        raise ValueError("KB file path is required.")
   
   

    # Create RAG model
    rag = RAGModel(kb_path=kb_path, google_api_key=google_key)

    # Requirement + options
    user_req = input("Enter the requirement (e.g., 'Unified Login', 'RBAC', or a freeform description): ").strip()
    include_neg = input("Include negative scenarios? (y/n): ").strip().lower().startswith("y")

    # Enable simple corrective RAG
    use_corrective = input("Enable corrective RAG (auto revalidate & regenerate)? (y/n): ").strip().lower().startswith("y")
    max_loops_str = input("Max corrective loops (default 3): ").strip()
    try:
        max_loops = int(max_loops_str) if max_loops_str else 3
    except ValueError:
        max_loops = 3

    # First pass (with or without corrective loop)
    if use_corrective:
        draft = rag.corrective_generate(
            user_req=user_req,
            include_neg=include_neg,
            initial_refinements="",
            max_loops=max_loops,
        )
    else:
        draft = rag.generate(user_req=user_req, include_neg=include_neg, refinements="")

    print("\n--- Draft Output ---\n")
    print(draft)

    # Optional manual refinement loop (still uses corrective mode if enabled)
    refine = input("\nProvide any refinements (press Enter to accept as-is): ").strip()
    if refine:
        if use_corrective:
            final = rag.corrective_generate(
                user_req=user_req,
                include_neg=include_neg,
                initial_refinements=refine,
                max_loops=max_loops,
            )
        else:
            final = rag.generate(user_req=user_req, include_neg=include_neg, refinements=refine)
        print("\n=== Final Output ===\n")
        print(final)
    else:
        print("\n=== Final Output ===\n")
        print(draft)

    print("\nDone.")

if __name__ == "__main__":
    main()