# DocuMind — LLM Response Validation Guide

> **Purpose:** For each test file ingested into DocuMind, this document lists 5 questions
> and their expected answers. Use these to manually or programmatically verify the LLM
> is retrieving and reasoning correctly — not hallucinating.
>
> **Pass criteria:** The LLM answer must contain the key facts listed under each question.
> Exact wording is not required — semantic match is sufficient.
>
> **Fail criteria:** Missing key facts, wrong numbers/dates, or fabricated information
> not present in the source document.

---

## How to Use This File

1. Ingest all test files via `POST /ingest`
2. Send each question via `POST /query` with the corresponding `doc_id`
3. Compare the LLM answer against the **Expected Key Facts**
4. Mark each as ✅ Pass or ❌ Fail

```bash
# Example query command
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "<question here>", "doc_ids": ["<doc_id>"]}'
```

---

---

## 📄 File 1: `contract_terms.txt`

**Document:** Service Agreement — Acme Corp & TechFlow Ltd
**Type:** Plain text
**Topics:** Payment terms, termination, confidentiality, liability, governing law

---

### Q1: What are the payment terms for invoices?

**Question to send:**
```
What are the payment terms for invoices?
```

**Expected Key Facts (LLM answer must include):**
- Invoices are due within **NET 30 days** of the invoice date
- Late payments incur a penalty of **1.5% per month**
- Payment must be made via **bank transfer**

**Expected answer should look like:**
> Invoices are due within NET 30 days of the invoice date. Late payments will incur a
> penalty of 1.5% per month on outstanding balances. Payments must be made via bank transfer.

**Red flags (fail if present):**
- Says NET 60 or any number other than 30
- Omits the late payment penalty
- Mentions payment by cheque or credit card

---

### Q2: What happens if either party wants to terminate the agreement?

**Question to send:**
```
What is the termination clause — how much notice is required?
```

**Expected Key Facts:**
- Either party may terminate with **30 days written notice**
- Immediate termination is allowed in cases of **material breach**
- Upon termination, all **unpaid invoices become immediately due**

**Expected answer should look like:**
> Either party may terminate the agreement with 30 days written notice. Immediate
> termination is permitted in cases of material breach. Upon termination, all unpaid
> invoices become immediately due and payable.

**Red flags:**
- States a notice period other than 30 days
- Does not mention the material breach exception
- Does not mention the immediate payment of unpaid invoices

---

### Q3: How long does the confidentiality obligation last?

**Question to send:**
```
How long does the confidentiality obligation survive after termination?
```

**Expected Key Facts:**
- Confidentiality obligations survive termination for **3 years**
- Violation may result in damages of up to **$50,000 USD**

**Expected answer should look like:**
> The confidentiality obligations survive termination for a period of 3 years.
> Violation of confidentiality may result in damages of up to $50,000 USD.

**Red flags:**
- States any duration other than 3 years
- Omits the $50,000 penalty
- Says confidentiality ends at termination

---

### Q4: What is the maximum liability cap under this contract?

**Question to send:**
```
What is the maximum liability cap in the contract?
```

**Expected Key Facts:**
- Maximum liability is capped at the **total contract value of $120,000 USD**
- Neither party is liable for **indirect or consequential damages**

**Expected answer should look like:**
> The maximum liability is capped at the total contract value of $120,000 USD.
> Neither party is liable for indirect or consequential damages.

**Red flags:**
- States a different dollar amount
- Says there is no liability cap
- Omits the exclusion of consequential damages

---

### Q5: Which law governs the contract and where are disputes resolved?

**Question to send:**
```
Which governing law applies to this agreement and where are disputes resolved?
```

**Expected Key Facts:**
- Governed by the laws of the **State of California, USA**
- Disputes resolved by **binding arbitration**
- Arbitration location: **San Francisco, CA**

**Expected answer should look like:**
> This agreement is governed by the laws of the State of California, USA.
> Disputes shall be resolved by binding arbitration in San Francisco, California.

**Red flags:**
- States a different jurisdiction (e.g., New York, Delaware)
- Says disputes go to court rather than arbitration
- Omits the San Francisco location

---

---

## 📝 File 2: `architecture_guide.md`

**Document:** DocuMind Technical Architecture Guide
**Type:** Markdown
**Topics:** RAG pipeline layers, FlashRank, API endpoints, configuration, benchmarks

---

### Q1: What are the three layers of the retrieval pipeline?

**Question to send:**
```
What are the three layers of the DocuMind retrieval pipeline?
```

**Expected Key Facts:**
- Layer 1: **Hybrid Retrieval** — Dense (ChromaDB) + Sparse (BM25) fused via RRF
- Layer 2: **Re-Ranking** — FlashRank cross-encoder (`ms-marco-MiniLM-L-12-v2`)
- Layer 3: **Confidence Filtering** — drops chunks below threshold, returns "I don't know"

**Expected answer should look like:**
> The three layers are: (1) Hybrid Retrieval combining dense semantic search via ChromaDB
> and sparse BM25 keyword matching using Reciprocal Rank Fusion, (2) Re-Ranking using
> FlashRank with the ms-marco-MiniLM-L-12-v2 cross-encoder model, and (3) Confidence
> Filtering which drops low-scoring chunks and prevents hallucination.

**Red flags:**
- Describes fewer than 3 layers
- Calls the re-ranker something other than FlashRank
- Omits the anti-hallucination purpose of confidence filtering

---

### Q2: What embedding model is used and what are the dense/sparse retrieval weights?

**Question to send:**
```
What embedding model is used for dense retrieval and what are the RRF fusion weights?
```

**Expected Key Facts:**
- Embedding model: **nomic-embed-text** (via Ollama)
- Dense weight: **0.6** (60%)
- Sparse (BM25) weight: **0.4** (40%)

**Expected answer should look like:**
> Dense retrieval uses the nomic-embed-text embedding model via Ollama. The RRF fusion
> weights are 0.6 for dense retrieval and 0.4 for sparse BM25 retrieval.

**Red flags:**
- Names a different embedding model (e.g., text-embedding-ada-002)
- States incorrect weights (e.g., 50/50 or reversed)
- Does not mention BM25

---

### Q3: What does the POST /ingest endpoint return?

**Question to send:**
```
What does the POST /ingest API endpoint return on success?
```

**Expected Key Facts:**
- Returns HTTP **201**
- Response contains: `doc_id`, `status: "ready"`, `chunk_count`

**Expected answer should look like:**
> On success, POST /ingest returns HTTP 201 with a JSON body containing the doc_id
> (a UUID), a status field set to "ready", and the chunk_count indicating how many
> chunks were stored.

**Red flags:**
- Says HTTP 200 instead of 201
- Omits the `doc_id` field
- Does not mention `chunk_count`

---

### Q4: What is the average query latency with and without re-ranking?

**Question to send:**
```
What is the average query latency with re-ranking enabled versus disabled?
```

**Expected Key Facts:**
- With re-ranking: **~2.1 seconds**
- Without re-ranking: **~0.9 seconds**

**Expected answer should look like:**
> According to the performance benchmarks, the average query latency with re-ranking
> enabled is approximately 2.1 seconds, while without re-ranking it is approximately
> 0.9 seconds.

**Red flags:**
- Swaps the two values
- States significantly different numbers
- Says latency is the same with and without re-ranking

---

### Q5: What are the known limitations of DocuMind?

**Question to send:**
```
What are the known limitations of the DocuMind system?
```

**Expected Key Facts (at least 3 of 4 must be mentioned):**
- **Scanned PDFs** require OCR pre-processing (not automatic)
- **Tables in PDFs** may lose formatting during extraction
- **Changing embedding models** requires full re-ingestion
- **BM25 index** is rebuilt on every query (cache recommended for >10k chunks)

**Expected answer should look like:**
> Known limitations include: scanned PDFs requiring manual OCR pre-processing, tables
> in PDFs potentially losing formatting during extraction, the need to re-ingest all
> documents if the embedding model is changed, and the BM25 index being rebuilt on
> every query which can be slow for large collections over 10,000 chunks.

**Red flags:**
- Lists fewer than 3 limitations
- Fabricates limitations not in the document
- Does not mention OCR or re-ingestion requirements

---

---

## 📘 File 3: `employee_handbook.docx`

**Document:** Employee Handbook — TechFlow Ltd v3.1
**Type:** Word document (.docx)
**Topics:** Leave policy, benefits, salary, code of conduct, remote work, IT security

---

### Q1: How many days of annual leave are employees entitled to?

**Question to send:**
```
How many days of annual leave do employees receive, and does it increase over time?
```

**Expected Key Facts:**
- **20 days** of annual leave per year
- Increases to **25 days** after **5 years** of service

**Expected answer should look like:**
> Employees are entitled to 20 days of annual leave per calendar year. This increases
> to 25 days after 5 years of service.

**Red flags:**
- States a different number of leave days
- Does not mention the increase after 5 years
- Confuses annual leave with sick leave

---

### Q2: What is the parental leave policy?

**Question to send:**
```
What parental leave is offered to primary caregivers?
```

**Expected Key Facts:**
- **16 weeks** of parental leave
- **Fully paid** for primary caregivers

**Expected answer should look like:**
> Primary caregivers are entitled to 16 weeks of fully paid parental leave.

**Red flags:**
- States a different duration (e.g., 12 weeks or 6 months)
- Says parental leave is unpaid or partially paid
- Does not distinguish primary caregiver entitlement

---

### Q3: What is the remote work policy including core hours?

**Question to send:**
```
What is the remote work policy — how many days are allowed and what are the core hours?
```

**Expected Key Facts:**
- Up to **3 days per week** from home
- Available after **90 days** of employment
- Core hours: **10:00 AM to 3:00 PM Pacific Time**
- VPN is **mandatory** when accessing company systems remotely

**Expected answer should look like:**
> Employees may work remotely up to 3 days per week after completing 90 days of
> employment. Remote workers must be available during core hours of 10:00 AM to
> 3:00 PM Pacific Time, and VPN usage is mandatory when accessing company systems.

**Red flags:**
- States a different number of remote days
- Omits the 90-day waiting period
- Does not mention the VPN requirement

---

### Q4: What is the annual learning and development budget?

**Question to send:**
```
How much is the annual learning and development budget for employees?
```

**Expected Key Facts:**
- **$2,000 per year**
- Can be used for **courses, conferences, and books**

**Expected answer should look like:**
> Employees receive a learning and development budget of $2,000 per year, which can
> be used for courses, conferences, and books.

**Red flags:**
- States a different dollar amount
- Says the budget is only for one type of use (e.g., only courses)
- Confuses learning budget with another benefit

---

### Q5: What IT security requirements must employees follow?

**Question to send:**
```
What are the IT security requirements employees must comply with?
```

**Expected Key Facts (all 4 must be present):**
- Passwords at least **16 characters**, rotated every **90 days**
- **MFA (Multi-Factor Authentication)** mandatory for all company accounts
- Security incidents reported to **security@techflow.com** within **2 hours**
- Software installation requires **IT department approval**

**Expected answer should look like:**
> Employees must comply with the following IT security requirements: passwords must
> be at least 16 characters and rotated every 90 days; MFA is mandatory for all
> company accounts; security incidents must be reported to security@techflow.com
> within 2 hours of detection; and software installation requires prior IT approval.

**Red flags:**
- States password length other than 16 characters
- Says 90 days for rotation but gets it wrong
- Omits the 2-hour incident reporting window
- States a different reporting email address

---

---

## 📕 File 4: `product_manual.pdf`

**Document:** CloudSync Pro User Manual v2.4
**Type:** PDF
**Topics:** System requirements, installation, pricing plans, troubleshooting, support

---

### Q1: What are the minimum system requirements for Windows?

**Question to send:**
```
What are the minimum system requirements to run CloudSync Pro on Windows?
```

**Expected Key Facts:**
- **Windows 10 (64-bit)** minimum
- **4 GB RAM** minimum
- Recommended: **Windows 11**, **8 GB RAM**, **SSD**

**Expected answer should look like:**
> The minimum system requirements for Windows are Windows 10 (64-bit) with 4 GB of
> RAM. The recommended configuration is Windows 11 with 8 GB of RAM and an SSD.

**Red flags:**
- States Windows 7 or Windows 8 as minimum
- States wrong RAM requirement
- Confuses minimum with recommended specs

---

### Q2: What are the pricing plans and how much does the Professional plan cost?

**Question to send:**
```
What are the pricing plans for CloudSync Pro and what does the Professional plan include?
```

**Expected Key Facts:**
- Professional plan costs **$29/month**
- Includes **1 TB** of storage
- Supports up to **5 users**
- Includes **priority email** support

**Expected answer should look like:**
> CloudSync Pro offers four plans: Starter ($9/mo), Professional ($29/mo), Business
> ($79/mo), and Enterprise (custom pricing). The Professional plan includes 1 TB of
> storage for up to 5 users with priority email support.

**Red flags:**
- States incorrect price for Professional plan
- Confuses storage between plans (e.g., says 100 GB for Professional)
- Lists incorrect number of users
- Omits any plans from the table

---

### Q3: How do you install CloudSync Pro on macOS?

**Question to send:**
```
How do I install CloudSync Pro on macOS?
```

**Expected Key Facts:**
- Download the **DMG file** from the website
- Drag **CloudSync Pro to Applications** folder
- macOS may require approval under **System Settings > Privacy & Security**

**Expected answer should look like:**
> To install on macOS, download the CloudSyncPro-2.4.dmg file from the website.
> Open the DMG and drag CloudSync Pro to your Applications folder. On first launch,
> macOS may prompt you to approve the app under System Settings > Privacy & Security.

**Red flags:**
- Describes Windows installation steps instead
- Omits the Privacy & Security approval step
- Says to run an .exe file

---

### Q4: How do I fix files stuck in "Pending" sync state?

**Question to send:**
```
How do I troubleshoot files that are stuck in pending sync state?
```

**Expected Key Facts:**
- Use **File > Force Full Sync** to force sync
- Clear the **local cache** at `~/.cloudsync/cache/` if problem persists
- **Restart** CloudSync Pro after clearing cache

**Expected answer should look like:**
> To fix files stuck in pending state, use File > Force Full Sync. If the problem
> persists, clear the local cache located at ~/.cloudsync/cache/ and then restart
> CloudSync Pro.

**Red flags:**
- Describes a fix for a different issue (e.g., high CPU or auth errors)
- Does not mention the cache directory path
- Omits the Force Full Sync step

---

### Q5: What should I do if CloudSync Pro is using too much CPU?

**Question to send:**
```
CloudSync Pro is using very high CPU. What are the recommended fixes?
```

**Expected Key Facts:**
- Reduce **sync frequency** to every 15 minutes
- Exclude large binary folders such as **node_modules** and **.git** from sync

**Expected answer should look like:**
> To reduce high CPU usage, reduce the sync frequency to every 15 minutes in settings.
> Additionally, exclude large binary folders such as node_modules and .git from the
> sync scope.

**Red flags:**
- Recommends reinstalling the application
- Does not mention sync frequency adjustment
- Does not mention excluding specific folders like node_modules

---

---

## 🌐 File 5: Web URL (Wikipedia — Retrieval-Augmented Generation)

**Source:** `https://en.wikipedia.org/wiki/Retrieval-augmented_generation`
**Type:** Web URL ingestion
**Topics:** RAG definition, components, applications, history

---

### Q1: What is retrieval-augmented generation (RAG)?

**Question to send:**
```
What is retrieval-augmented generation (RAG)?
```

**Expected Key Facts:**
- RAG is a technique that **combines a retrieval system with a language model**
- It retrieves **relevant documents or passages** from an external knowledge base
- The retrieved content is used to **augment the LLM's prompt**
- Improves accuracy by grounding answers in **factual, external knowledge**

**Expected answer should look like:**
> Retrieval-Augmented Generation (RAG) is a technique that combines a large language
> model with a retrieval system. When a query is made, relevant documents are retrieved
> from an external knowledge base and provided as context to the language model,
> improving the accuracy and factual grounding of its responses.

**Red flags:**
- Defines RAG as a standalone LLM with no retrieval component
- Confuses RAG with fine-tuning
- Does not mention the external knowledge base

---

### Q2: Why is RAG preferable to fine-tuning for knowledge updates?

**Question to send:**
```
Why is RAG preferred over fine-tuning when the knowledge base needs to be updated frequently?
```

**Expected Key Facts:**
- Fine-tuning requires **retraining the model**, which is expensive and slow
- RAG allows knowledge to be **updated by changing the document store** without retraining
- RAG is more **cost-effective** for dynamic or frequently changing knowledge

**Expected answer should look like:**
> RAG is preferred over fine-tuning for frequently updated knowledge because it allows
> the knowledge base to be updated simply by modifying the document store, without
> retraining the language model. Fine-tuning requires expensive and time-consuming
> model retraining each time the knowledge changes.

**Red flags:**
- Says fine-tuning is faster than RAG for knowledge updates
- Does not mention the cost or time of retraining
- Claims RAG and fine-tuning are equivalent approaches

---

### Q3: What are the main components of a RAG system?

**Question to send:**
```
What are the main components that make up a RAG system?
```

**Expected Key Facts (at least 3 must be mentioned):**
- A **document corpus / knowledge base**
- A **retriever** (e.g., dense vector search or BM25)
- A **language model** (generator)
- An **embedding model** for encoding documents and queries

**Expected answer should look like:**
> A RAG system consists of a document corpus or knowledge base, a retriever that
> finds relevant documents using techniques such as dense vector search or BM25,
> an embedding model for encoding documents and queries, and a language model that
> generates the final answer conditioned on the retrieved context.

**Red flags:**
- Omits the retriever component
- Does not mention embeddings
- Describes a system with only an LLM and no retrieval

---

### Q4: What problem does RAG solve that standard LLMs have?

**Question to send:**
```
What problem with standard LLMs does RAG help to solve?
```

**Expected Key Facts:**
- LLMs have a **knowledge cutoff date** and cannot access recent information
- LLMs can **hallucinate** — generating plausible but factually incorrect answers
- RAG grounds answers in **verifiable, up-to-date external documents**

**Expected answer should look like:**
> RAG addresses two key problems with standard LLMs: first, LLMs have a fixed
> knowledge cutoff and cannot access information beyond their training data; second,
> LLMs can hallucinate plausible but incorrect information. RAG mitigates both by
> grounding responses in retrieved, verifiable external documents.

**Red flags:**
- Does not mention hallucination
- Does not mention the knowledge cutoff limitation
- Claims RAG eliminates all LLM errors

---

### Q5: In what domains or applications is RAG commonly used?

**Question to send:**
```
What are common real-world applications or domains where RAG is used?
```

**Expected Key Facts (at least 2 must be mentioned):**
- **Question answering** over private documents or enterprise knowledge bases
- **Customer support** chatbots with access to product documentation
- **Legal or medical** document analysis
- **Code assistants** with access to codebases or documentation

**Expected answer should look like:**
> RAG is commonly used in enterprise question answering systems over private document
> collections, customer support chatbots grounded in product documentation, legal and
> medical document analysis, and AI code assistants that can retrieve relevant
> documentation or code snippets.

**Red flags:**
- Only lists generic LLM applications with no retrieval-specific context
- Claims RAG is only used in one domain
- Lists applications that are clearly unrelated to document retrieval

---

---

## ✅ Test Run Checklist

Use this table to track your test results:

| # | File | Question | Pass / Fail | Notes |
|---|------|----------|-------------|-------|
| 1.1 | contract_terms.txt | Payment terms | | |
| 1.2 | contract_terms.txt | Termination notice | | |
| 1.3 | contract_terms.txt | Confidentiality duration | | |
| 1.4 | contract_terms.txt | Liability cap | | |
| 1.5 | contract_terms.txt | Governing law | | |
| 2.1 | architecture_guide.md | Three retrieval layers | | |
| 2.2 | architecture_guide.md | Embedding model + RRF weights | | |
| 2.3 | architecture_guide.md | POST /ingest response | | |
| 2.4 | architecture_guide.md | Query latency benchmarks | | |
| 2.5 | architecture_guide.md | Known limitations | | |
| 3.1 | employee_handbook.docx | Annual leave days | | |
| 3.2 | employee_handbook.docx | Parental leave | | |
| 3.3 | employee_handbook.docx | Remote work policy | | |
| 3.4 | employee_handbook.docx | Learning budget | | |
| 3.5 | employee_handbook.docx | IT security requirements | | |
| 4.1 | product_manual.pdf | Windows system requirements | | |
| 4.2 | product_manual.pdf | Pricing plans | | |
| 4.3 | product_manual.pdf | macOS installation | | |
| 4.4 | product_manual.pdf | Pending sync fix | | |
| 4.5 | product_manual.pdf | High CPU fix | | |
| 5.1 | URL (Wikipedia/RAG) | RAG definition | | |
| 5.2 | URL (Wikipedia/RAG) | RAG vs fine-tuning | | |
| 5.3 | URL (Wikipedia/RAG) | RAG components | | |
| 5.4 | URL (Wikipedia/RAG) | Problems RAG solves | | |
| 5.5 | URL (Wikipedia/RAG) | RAG applications | | |

**Total: 25 questions across 5 file types**

---

*Generated for DocuMind v0.1.0 — update expected answers if source documents are modified.*
