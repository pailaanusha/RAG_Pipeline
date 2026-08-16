
# AI-Powered User Story & Acceptance Criteria Generator (RAG + Recursive Split + FAISS) 
# - KB: .docx (e.g., /content/RAG_ready_requirements_rules.docx)
# - Chunking: RecursiveCharacterTextSplitter
# - Vector DB: FAISS
# - Embeddings: Google Gemini (text-embedding-004)
# - LLM: ChatGoogleGenerativeAI (gemini-1.5-flash)
# - Interaction: input() prompts
# ----------------------------------------------------------

import os
from typing import List

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
            model="gemini-flash-latest",
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
            separators=["\n## ", "\n### ", "\n", ". ", " ", ""],  # recursive coarse->fine
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
- 2–3 stories per feature; each ≤ 120 words.
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
        return PromptTemplate(
            template=template,
            input_variables=["user_req"],
        )

    def generate(self, user_req: str, include_neg: bool, refinements: str = "") -> str:
        kb_context = self._retrieve(user_req)
        prompt = self._build_prompt(user_req, kb_context, include_neg, refinements)
        chain = prompt | self.llm
        result = chain.invoke({"user_req": user_req})
        return result.content

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

    # First pass
    draft = rag.generate(user_req=user_req, include_neg=include_neg, refinements="")
    print("\n--- Draft Output ---\n")
    print(draft)

    # Refinement loop
    refine = input("\nProvide any refinements (press Enter to accept as-is): ").strip()
    if refine:
        final = rag.generate(user_req=user_req, include_neg=include_neg, refinements=refine)
        print("\n=== Final Output ===\n")
        print(final)
    else:
        print("\n=== Final Output ===\n")
        print(draft)

    print("\nDone.")

if __name__ == "__main__":
    main()
