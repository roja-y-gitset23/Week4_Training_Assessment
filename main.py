"""FastAPI app: POST /ask, POST /classify, POST /chat (model chooses tools or RAG)."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from google.genai import types
from pydantic import BaseModel

import rag
from tool_calling import declarations as BASE_DECLARATIONS  # Task 3 tools
from tool_calling import run_tool as run_basic_tool


@asynccontextmanager
async def lifespan(app):
    rag.get_collection()  # builds the index on first start if it does not exist
    yield


app = FastAPI(title="RAG Assistant API", lifespan=lifespan)


class AskIn(BaseModel):
    question: str


class ClassifyIn(BaseModel):
    text: str


class ChatIn(BaseModel):
    message: str


# ------------------------------------------------------------------ /ask
@app.post("/ask")
def ask(body: AskIn):
    return rag.answer_question(body.question)


# ------------------------------------------------------------------ /classify
CATEGORIES = ["Billing", "Technical", "Account"]
FALLBACK_PROMPT = (
    "You are a support-ticket classifier. Classify the ticket into exactly one of: "
    "Billing, Technical, Account. Reply with the category name only."
)


@app.post("/classify")
def classify(body: ClassifyIn):
    path = Path("prompts/classifier_v2.txt")  # best prompt from the Prompt Engineering task
    prompt = path.read_text(encoding="utf-8") if path.exists() else FALLBACK_PROMPT
    if "{ticket}" in prompt:
        contents, system = prompt.replace("{ticket}", body.text), None
    else:
        contents, system = body.text, prompt
    resp = rag.with_retry(lambda: rag.client.models.generate_content(
        model=rag.MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system, temperature=0),
    ))
    raw = (resp.text or "").strip()
    category = next((c for c in CATEGORIES if c.lower() in raw.lower()), raw)
    return {"category": category}


# ------------------------------------------------------------------ /chat
search_manual_decl = types.FunctionDeclaration(
    name="search_manual",
    description="Search the product manual for information. Use for any product/how-to question.",
    parameters=types.Schema(
        type="OBJECT",
        properties={"question": types.Schema(type="STRING", description="Question to search for")},
        required=["question"],
    ),
)

CHAT_SYSTEM = (
    "You are a support assistant. Use get_customer for customer/plan questions, "
    "get_order_status for order questions, and search_manual for product manual questions. "
    "Answer ONLY from tool results. If search_manual does not contain the answer, "
    "reply exactly: I don't know. If a tool reports an id not found, say so. "
    "Keep the final answer to one or two short sentences."
)

chat_config = types.GenerateContentConfig(
    system_instruction=CHAT_SYSTEM,
    tools=[types.Tool(function_declarations=BASE_DECLARATIONS + [search_manual_decl])],
    temperature=0,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


@app.post("/chat")
def chat(body: ChatIn):
    contents = [types.Content(role="user", parts=[types.Part(text=body.message)])]
    tools_used, pages = [], set()
    answer = "No answer produced."

    for _ in range(5):
        resp = rag.with_retry(lambda: rag.client.models.generate_content(
            model=rag.MODEL, contents=contents, config=chat_config))
        calls = resp.function_calls
        if not calls:
            answer = (resp.text or "").strip().replace("\u2019", "'")
            break

        contents.append(resp.candidates[0].content)
        parts = []
        for call in calls:
            args = dict(call.args or {})
            tools_used.append(call.name)
            if call.name == "search_manual":
                chunks = rag.retrieve(str(args.get("question", "")))
                pages.update(c["page"] for c in chunks)
                result = {"ok": True, "chunks": [{"page": c["page"], "text": c["text"]} for c in chunks]}
            else:
                result = run_basic_tool(call.name, args)  # unknown tool name handled safely
            parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=parts))

    sources = [{"page": p} for p in sorted(pages)] if "search_manual" in tools_used else []
    if answer.lower().startswith("i don't know"):
        answer, sources = "I don't know.", []
    return {"answer": answer, "tools_used": tools_used, "sources": sources}