# Week_4 Training Assessment

Four LLM engineering tasks built with Python and the Gemini API (free tier).

| # | Task | Main files |
|---|------|-----------|
| 1 | Prompt Engineering and Golden Dataset | `prompts/classifier_v1.txt`, `prompts/classifier_v2.txt`, `golden.jsonl`, `eval.py` |
| 2 | Structured Output with Validation | `extract.py` |
| 3 | Tool Calling | `tool_calling.py` |
| 4 | RAG Assistant API | `rag.py`, `main.py`, `eval_rag.py` |

## Setup

```powershell
cd F:\Week4_Training_Assessment
python -m venv venv
venv\Scripts\Activate.ps1
pip install google-genai python-dotenv fastapi uvicorn pydantic httpx pypdf langchain-text-splitters chromadb
pip freeze > requirements.txt
```

Create a `.env` file (never commit it):

```
GEMINI_API_KEY=gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

The free tier allows about 15 requests per minute. If you see a `429` error, wait a minute and run again.

---

## Task 1: Prompt Engineering and Golden Dataset

Classifies support tickets into **Billing**, **Technical** or **Account**.

- `prompts/classifier_v1.txt`: simple baseline prompt.
- `prompts/classifier_v2.txt`: improved prompt with all six parts: Role, Context, Task, Output format, Constraints, Examples.
- `golden.jsonl`: 12 labelled ticket/category pairs.
- `run_eval(version)`: runs a prompt version over the golden set at temperature 0 and prints accuracy.
- Each prompt version is committed separately so the v1 vs v2 difference is visible in git history.

```powershell
python eval.py
```

Expected output format: `v1: 8/12 (66%) | v2: 11/12 (91%)`

## Task 2: Structured Output with Validation

Extracts a Pydantic `TicketInfo` (customer_name, product, issue_type, urgency 1-5, summary) from 5 free-text tickets.

- Uses JSON mode and `model_validate_json`.
- On a `ValidationError`, the error is sent back to the model and the call is retried, up to 3 attempts.
- After 3 failed attempts a clear failure result is returned instead of crashing.

```powershell
python extract.py
```
Expected output format:
```
(venv) PS C:\Week4_Training_Assessment> python extract.py

Ticket 1:
Attempt 1: success

Ticket 2:
Attempt 1: success

Ticket 3:
Attempt 1: validation error (urgency must be 1-5)
Attempt 2: success

Ticket 4:
Attempt 1: success

Ticket 5:
Attempt 1: success

Results:

Ticket 1: {'customer_name': 'Priya Sharma', 'product': 'CloudSync', 'issue_type': 'technical', 'urgency': 1, 'summary': "The customer's CloudSync desktop app crashes upon opening."}
Ticket 2: {'customer_name': 'Daniel Wu', 'product': 'BillPro', 'issue_type': 'billing', 'urgency': 2, 'summary': 'Daniel Wu was double charged for his Pro plan on BillPro this month.'}
Ticket 3: {'customer_name': 'Maria Lopez', 'product': 'TeamChat', 'issue_type': 'account', 'urgency': 5, 'summary': 'The entire team is locked out of TeamChat right before an urgent client demo.'}
Ticket 4: {'customer_name': 'Ahmed Khan', 'product': 'DataViz', 'issue_type': 'how-to', 'urgency': 1, 'summary': 'The customer is asking for instructions on how to export reports from DataViz with no urgency.'}
Ticket 5: {'customer_name': "Sarah O'Neil", 'product': 'BillPro', 'issue_type': 'technical', 'urgency': 3, 'summary': 'The invoice PDF in BillPro displays a blank page and needs to be resolved by the end of the week.'}
```
python extract.py 3
Attempt 1: validation error (urgency must be 1-5)
Attempt 2: success

Results:

Ticket 3: {'customer_name': 'Maria Lopez', 'product': 'TeamChat', 'issue_type': 'technical', 'urgency': 5, 'summary': 'The entire team is locked out of TeamChat right before a crucial client demo.'}

## Task 3: Tool Calling
Two tools the model chooses between:

- `get_customer(customer_id)` reads a mock CRM dictionary.
- `get_order_status(order_id)` reads a small in-memory SQLite `orders` table.

Tools return structured dictionaries such as `{"ok": True, ...}` or `{"ok": False, "error": "..."}`. Unknown ids and non-existent tool names are handled safely without crashing.

```powershell
python tool_calling.py
```
Output:
=== Main test ===
Tool: get_customer(2) | Tool: get_order_status(1042)
Answer: Customer 2 is on the Premium plan and order 1042 is out for delivery.

The script also tests an unknown id (customer 99, order 5555) and a non-existent tool name (`delete_customer`).

=== Unknown id ===
Tool: get_customer(99) | Tool: get_order_status(5555)
Answer: Customer 99 was not found and order 5555 was not found.

=== Non-existent tool name (called directly) ===
{'ok': False, 'error': "Unknown tool 'delete_customer'"}

## Task 4: RAG Assistant API

FastAPI service that answers questions from a product manual PDF.

**How it works**

1. Load `manual.pdf` page by page with `pypdf`.
2. Split with `RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)`.
3. Embed with Gemini embeddings and store in Chroma (`chroma_db/`) with `page` metadata.
4. Retrieve the top 3 chunks per question.
5. Answer only from that context using an anti-hallucination prompt. If the context lacks the answer, the reply is exactly `I don't know.`

**Endpoints**

| Endpoint | Purpose |
|----------|---------|
| `POST /ask` | RAG answer with page sources |
| `POST /classify` | Ticket classification using the best prompt (`classifier_v2.txt`) |
| `POST /chat` | The model chooses between `get_customer`, `get_order_status` and `search_manual` (RAG) |

**Run**

```powershell
python rag.py ingest          # build the index once
uvicorn main:app --reload     # start the API, docs at http://127.0.0.1:8000/docs
```

**Example**

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json" -Body '{"question":"How do I reset the device?"}'
```

- Answerable: `{"answer": "...", "sources": [{"page": 7}]}`
- Unanswerable: `{"answer": "I don't know.", "sources": []}`

**Evaluation and failure analysis**

```powershell
python eval_rag.py
```

Runs 6 answerable and 4 unanswerable questions and classifies every wrong answer:

- **Retrieval failure:** the answer is not in the top 3 chunks.
- **Generation failure:** the answer is in the chunks but the model got it wrong, or answered an unanswerable question.

| Question | Wrong answer | Cause | Fix applied | Result after fix |
|----------|--------------|-------|-------------|------------------|
|          |              |       |             |                  |

---

## Project structure

```
F:\Week4_Training_Assessment
├── prompts/
│   ├── classifier_v1.txt
│   └── classifier_v2.txt
├── golden.jsonl
├── eval.py
├── extract.py
├── tool_calling.py
├── rag.py
├── main.py
├── eval_rag.py
├── manual.pdf
├── requirements.txt
├── .env            
└── chroma_db/      
```

## Notes

- Secrets live in `.env`, and `.env`, `chroma_db/` and `venv/` are listed in `.gitignore`.