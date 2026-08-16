# AI-Powered RAG for Requirements Engineering & QA

**Basic RAG • Corrective RAG • Knowledge-Enriched RAG + Test Case Generation**

## 1. Overview

This project demonstrates three progressively enhanced Retrieval-Augmented Generation (RAG) implementations for AI-assisted requirements engineering and software testing.

1. **Basic RAG Pipeline** – retrieves business and compliance knowledge and generates Agile User Stories and Acceptance Criteria.
2. **Corrective RAG** – adds automated validation, gap detection, retrieval/prompt refinement, and regeneration loops.
3. **Corrective RAG + KB Update + Test Generation** – stores generated User Stories in the Knowledge Base, rebuilds FAISS, and generates test cases for selected stories.

The project demonstrates how a RAG solution can evolve from simple grounded generation into a **feedback-driven, multi-stage requirements-to-testing workflow**.

---

## 2. Problem Statement

Traditional requirements analysis often requires manually converting:

- Business requirements
- Business rules
- Compliance requirements
- Security requirements
- Non-functional requirements

into:

- Agile User Stories
- Acceptance Criteria
- Gherkin scenarios
- QA Test Cases

This project explores how **RAG + LLMs** can assist that process while grounding generated artifacts in a domain-specific Knowledge Base.

---

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM | Google Gemini |
| Embedding Model | Google `text-embedding-004` |
| RAG Framework | LangChain |
| Vector Database | FAISS |
| Knowledge Base | Microsoft Word `.docx` |
| Document Loader | `Docx2txtLoader` |
| Text Splitter | `RecursiveCharacterTextSplitter` |
| Prompting | LangChain `PromptTemplate` |
| Acceptance Criteria | Gherkin / Given-When-Then |
| Environment | Google Colab / Python |

---

# 4. Common RAG Architecture

All three implementations use the same fundamental RAG foundation.

```text
Knowledge Base (.docx)
        |
        v
Document Loader
        |
        v
Recursive Character Text Splitter
        |
        v
Text Chunks
        |
        v
Gemini Embeddings
        |
        v
FAISS Vector Database
        |
        v
Retriever
        |
        v
User Requirement / Query
        |
        v
Relevant KB Context
        |
        v
Prompt + Retrieved Context
        |
        v
Gemini LLM
        |
        v
Generated Output
```

### Core RAG concept

```text
User Query
    ↓
Retrieve relevant organizational/domain knowledge
    ↓
Augment the prompt with retrieved context
    ↓
LLM generates a grounded response
```

---

# 5. Knowledge Base

The project uses a realistic synthetic **Digital Wallet & Secure Payments** Knowledge Base containing:

### Business Rules

- Transaction authorization
- Wallet balance validation
- Transaction limits
- Idempotency
- Atomicity
- Fraud/risk controls
- RBAC
- Refund and reversal rules
- Account closure
- Auditability

### Compliance and Security Rules

- PCI DSS
- GDPR
- CCPA/CPRA
- WCAG 2.1 AA
- Encryption
- Authentication and authorization
- Audit logging
- Data retention
- Security controls

### Non-Functional Requirements

- Performance
- Availability
- Reliability
- Scalability
- Observability
- Accessibility
- Security

The Knowledge Base acts as the **source of domain context** for the RAG pipeline.

---

# 6. Implementation 1 — Basic RAG Pipeline

## 6.1 Objective

The first implementation demonstrates a standard:

> **Retrieve → Augment → Generate**

For example, the user can enter:

```text
Peer-to-Peer Wallet Transfer
```

The system retrieves relevant business rules, compliance rules and non-functional requirements from the Knowledge Base.

Gemini then generates:

1. User Stories
2. Acceptance Criteria
3. Non-Functional Criteria
4. Standards Validation Summary

## 6.2 Architecture

```text
Digital Wallet Knowledge Base
          |
          v
      Load DOCX
          |
          v
   Recursive Chunking
          |
          v
    Gemini Embeddings
          |
          v
        FAISS
          |
          v
       Retriever
          |
          v
   User Requirement
          |
          v
  Relevant KB Context
          |
          v
   Prompt + Context
          |
          v
      Gemini LLM
          |
          v
 User Stories + Gherkin
 Acceptance Criteria
```

## 6.3 Processing Flow

```text
Load DOCX
   ↓
Split into chunks
   ↓
Generate embeddings
   ↓
Store embeddings in FAISS
   ↓
Accept user requirement
   ↓
Retrieve top relevant chunks
   ↓
Inject context into prompt
   ↓
Send prompt to Gemini
   ↓
Generate User Stories and Acceptance Criteria
```

### Important point

The first implementation has **no automated corrective loop**. If the LLM produces an incomplete response, the system does not automatically validate and regenerate it.

---

# 7. Implementation 2 — Corrective RAG

## 7.1 Objective

Basic RAG performs retrieval and generation, but generation quality is not automatically guaranteed.

The generated response might:

- Miss a required section
- Produce the wrong number of stories
- Miss Gherkin `Given/When/Then`
- Fail to mention relevant compliance requirements
- Miss negative scenarios
- Produce an overly long story

Corrective RAG introduces a **feedback loop**.

## 7.2 Architecture

```text
          Retrieve
             |
             v
          Generate
             |
             v
          Validate
             |
       +-----+-----+
       |           |
     PASS         FAIL
       |           |
       v           v
    Return    Identify Gaps
                   |
                   v
        Refine Prompt + Retrieval
                   |
                   v
              Retrieve Again
                   |
                   v
               Generate Again
                   |
                   +-------> Validate
```

The key difference is:

> **The system evaluates its generated output and uses the detected gaps to influence the next generation cycle.**

## 7.3 Validation Checks

The implementation validates:

- Required output sections
- 2–3 User Stories
- `As a / I want / so that` format
- Story length
- Gherkin `Given/When/Then`
- Compliance/security terminology
- Negative/edge scenarios when requested

## 7.4 Feedback Mechanism

Detected gaps are converted into refinement instructions.

Example:

```text
Please correct the following:

- Add Gherkin-style Acceptance Criteria.
- Include relevant compliance requirements.
- Include negative/edge scenarios.
```

The system can also generate additional retrieval terms such as:

```text
Gherkin
Acceptance Criteria
GDPR
CCPA
privacy
security baseline
error handling
edge cases
unauthorized
timeout
```

Therefore, corrective RAG modifies:

1. **Generation instructions**
2. **Retrieval query**

## 7.5 Iteration Control

The implementation supports a configurable maximum number of iterations.

```text
Generate
   ↓
Validate
   ↓
PASS → Return
   OR
FAIL → Identify gaps
   ↓
Refine prompt + retrieval terms
   ↓
Generate again
   ↓
Validate again
```

If the maximum number of loops is reached, the latest output is returned with potential gaps.

---

# 8. Implementation 3 — Corrective RAG + KB Update + Test Generation

## 8.1 Objective

The third implementation extends Corrective RAG into a **multi-stage requirements-to-testing workflow**.

The major new capability is:

> **Generated User Stories are added back to the Knowledge Base and the vector index is rebuilt before test-case generation.**

The workflow becomes:

```text
Requirement
     ↓
Corrective RAG
     ↓
Validated User Stories
     ↓
Update Knowledge Base
     ↓
Rebuild FAISS
     ↓
Retrieve Updated Knowledge
     ↓
Generate Test Cases
```

## 8.2 Architecture

```text
                 Original KB
                     |
                     v
              Chunk + Embed
                     |
                     v
                   FAISS
                     |
                     v
              User Requirement
                     |
                     v
              Corrective RAG
                     |
                     v
            Validated User Stories
                     |
                     v
              Extract Stories
                     |
                     v
               Update DOCX
                     |
                     v
              Rebuild FAISS
                     |
                     v
              Select User Story
                     |
                     v
          Retrieve Updated Context
                     |
                     v
             QA Test Generation
                     |
                     v
          Test Cases + Gherkin
```

---

# 9. Why Update the Knowledge Base?

Generated User Stories are not simply displayed and discarded.

They are appended to the original `.docx` Knowledge Base.

Conceptually:

```text
Original Knowledge
       +
Generated User Stories
       ↓
Enriched Knowledge Base
```

This creates a downstream knowledge source containing both:

- Original business/compliance knowledge
- Generated requirements artifacts

---

# 10. Why Rebuild FAISS?

Updating the `.docx` file does not automatically update the existing FAISS vector index.

Therefore:

```text
Updated DOCX
     ↓
Reload document
     ↓
Re-chunk
     ↓
Generate embeddings
     ↓
Create updated FAISS index
     ↓
Retriever can access new content
```

The implementation provides:

```python
rebuild_index()
```

to rebuild FAISS after the Knowledge Base is modified.

---

# 11. Test Case Generation

After the Knowledge Base is updated and FAISS is rebuilt, the user can select one or more generated User Stories.

The system performs another RAG stage:

```text
Selected User Story
        ↓
Retrieve relevant context
        ↓
Updated KB + FAISS
        ↓
QA Test Design Prompt
        ↓
Gemini
        ↓
Test Cases
```

Generated test cases contain:

- Test Case Title
- Preconditions
- Test Data
- Steps
- Expected Result
- Gherkin Scenarios
- Negative/edge scenarios when requested
- Relevant business, compliance and non-functional considerations

---

# 12. The Three Implementations Compared

| Capability | Basic RAG | Corrective RAG | Corrective + KB + Tests |
|---|:---:|:---:|:---:|
| Knowledge Base | ✓ | ✓ | ✓ |
| Document Chunking | ✓ | ✓ | ✓ |
| Embeddings | ✓ | ✓ | ✓ |
| FAISS Retrieval | ✓ | ✓ | ✓ |
| LLM Generation | ✓ | ✓ | ✓ |
| User Story Generation | ✓ | ✓ | ✓ |
| Gherkin Acceptance Criteria | ✓ | ✓ | ✓ |
| Automated Validation | — | ✓ | ✓ |
| Gap Detection | — | ✓ | ✓ |
| Feedback Loop | — | ✓ | ✓ |
| Retrieval Query Augmentation | — | ✓ | ✓ |
| Prompt Refinement | — | ✓ | ✓ |
| Knowledge Base Update | — | — | ✓ |
| FAISS Rebuild | — | — | ✓ |
| Second RAG Stage | — | — | ✓ |
| Test Case Generation | — | — | ✓ |

---

# 13. Conceptual Progression

## Level 1 — Basic RAG

```text
Knowledge
    ↓
Retrieval
    ↓
Generation
    ↓
User Stories
```

**Objective:** grounded generation.

## Level 2 — Corrective RAG

```text
Knowledge
    ↓
Retrieval
    ↓
Generation
    ↓
Validation
    ↓
Feedback
    ↓
Refinement
    ↓
Regeneration
```

**Objective:** improved output quality through automated correction.

## Level 3 — Multi-Stage Corrective RAG

```text
Requirement
    ↓
Corrective RAG
    ↓
Validated User Stories
    ↓
Update KB
    ↓
Rebuild FAISS
    ↓
Select User Story
    ↓
Retrieve Updated Context
    ↓
Generate Test Cases
```

**Objective:** connect requirements generation with downstream QA artifact generation.

---

# 14. Example End-to-End Scenario

Consider:

```text
Requirement:
Peer-to-Peer Wallet Transfer
```

### Stage 1 — Retrieval

The system retrieves rules related to:

- Sender authentication
- Recipient validation
- Balance
- Transfer limits
- Transaction authorization
- Idempotency
- Fraud detection
- Audit logging
- Privacy
- Security
- Accessibility
- Performance

### Stage 2 — User Story Generation

Gemini generates a User Story such as:

```text
As a wallet user, I want to transfer funds to another wallet,
so that I can send money securely to another user.
```

### Stage 3 — Acceptance Criteria

Example:

```text
Given the sender is authenticated
When the sender submits a valid transfer
Then the system should authorize and record the transaction
```

### Stage 4 — Corrective Validation

```text
Validation
    ↓
Gap Detection
    ↓
Refinement
    ↓
Retrieval
    ↓
Regeneration
```

### Stage 5 — Knowledge Base Update

```text
Original KB
    +
Generated User Stories
    ↓
Updated KB
```

### Stage 6 — FAISS Rebuild

The updated document is reloaded, chunked, embedded and indexed again.

### Stage 7 — Test Generation

The user selects a generated User Story.

```text
User Story
    ↓
Updated KB Retrieval
    ↓
QA Prompt
    ↓
Gemini
    ↓
Test Cases + Gherkin Scenarios
```

---

# 15. Recommended Repository Structure

```text
rag-requirements-qa/
│
├── README.md
│
├── basic_rag/
│   └── rag_user_story_generator.py
│
├── corrective_rag/
│   └── corrective_rag_user_story_generator.py
│
├── kb_update_test_generation/
│   └── rag_kb_update_test_generator.py
│
├── knowledge_base/
│   └── digital_wallet_business_compliance_rules.docx
│
├── requirements.txt
│
└── .gitignore
```

---

# 16. Security Considerations

API keys should never be hard-coded.

Use environment variables such as:

```text
GOOGLE_API_KEY
```

Recommended practices:

- Do not commit API keys.
- Use `.gitignore` for local secrets.
- Review generated requirements before treating them as authoritative.
- Do not upload confidential enterprise documents to an uncontrolled external LLM service.
- Apply appropriate access control and audit logging in production.
- Consider data retention and privacy requirements when using external LLM APIs.

---

# 17. Limitations

### Rule-Based Validation

The corrective validator mainly uses deterministic checks and regular expressions. It does not fully understand semantic correctness.

### Retrieval Quality

Top-k vector similarity does not guarantee that retrieved chunks are always the most useful.

### FAISS Rebuild

The third implementation rebuilds the entire index after KB modification. For a large production Knowledge Base, incremental indexing may be preferable.

### Generated Content

LLM-generated User Stories and Test Cases should be reviewed before being treated as authoritative artifacts.

---

# 18. Future Improvements

Potential production-level enhancements include:

- Hybrid retrieval using **Vector Search + BM25**
- Reranking
- Metadata filtering
- Document/version tracking
- Requirement IDs
- Traceability between User Stories and Test Cases
- Structured LLM output using schemas
- LLM-as-a-Judge evaluation
- Semantic validation
- Duplicate User Story detection
- Incremental FAISS updates
- Human approval workflow
- Automated API test generation
- Jira integration
- CI/CD integration
- Test execution and reporting
- Regression test optimization

A possible future architecture:

```text
Business Requirements
        ↓
RAG
        ↓
User Stories
        ↓
Corrective Validation
        ↓
Human Approval
        ↓
Knowledge Base
        ↓
RAG
        ↓
Test Cases
        ↓
API Automation
        ↓
Pytest
        ↓
CI/CD
        ↓
Test Results
        ↓
Regression Optimization
```

---

# 19. Key Concepts Demonstrated

- Retrieval-Augmented Generation
- Grounded LLM generation
- Document ingestion
- Recursive text chunking
- Embeddings
- Vector similarity search
- FAISS
- LangChain
- Prompt engineering
- Corrective RAG
- Feedback loops
- Output validation
- Retrieval query augmentation
- Knowledge-base enrichment
- Vector index rebuilding
- Multi-stage RAG
- Agile User Stories
- Gherkin Acceptance Criteria
- AI-assisted QA Test Case Generation
- Business-rule-aware generation
- Compliance-aware generation

---



---

# 20. Final Summary

### Basic RAG

```text
Knowledge Base
      ↓
Retrieval
      ↓
LLM
      ↓
User Stories + Acceptance Criteria
```

### Corrective RAG

```text
Knowledge Base
      ↓
Retrieval
      ↓
LLM
      ↓
Validation
      ↓
Gap Detection
      ↓
Prompt/Retrieval Refinement
      ↓
Regeneration
```

### Corrective RAG + KB Update + Test Generation

```text
Requirement
      ↓
Corrective RAG
      ↓
Validated User Stories
      ↓
Update Knowledge Base
      ↓
Rebuild FAISS
      ↓
Select User Story
      ↓
Retrieve Updated Context
      ↓
LLM QA Generation
      ↓
Test Cases + Gherkin Scenarios
```

## Overall Project Evolution

**Basic RAG → Corrective RAG → Knowledge-Enriched Multi-Stage RAG**

The three implementations demonstrate how an LLM application can evolve from simple retrieval-grounded generation into a feedback-driven and downstream-aware AI workflow for **requirements engineering and QA automation**.
