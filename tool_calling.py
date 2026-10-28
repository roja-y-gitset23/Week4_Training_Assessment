"""Task 3 - Tool Calling (Gemini, manual function-calling loop)."""
import os
import sqlite3

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------------------------------------------------------- data
CUSTOMERS = {  # mock CRM
    1: {"customer_id": 1, "name": "Asha Rao", "plan": "Basic"},
    2: {"customer_id": 2, "name": "Ravi Kumar", "plan": "Premium"},
    3: {"customer_id": 3, "name": "Meena Iyer", "plan": "Standard"},
}

db = sqlite3.connect(":memory:")  # small SQLite table
db.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, status TEXT NOT NULL)")
db.executemany(
    "INSERT INTO orders VALUES (?, ?)",
    [(1041, "delivered"), (1042, "out for delivery"), (1043, "processing")],
)
db.commit()


# ---------------------------------------------------------------- tools
def get_customer(customer_id: int) -> dict:
    customer = CUSTOMERS.get(customer_id)
    if customer is None:
        return {"ok": False, "error": f"Customer {customer_id} not found"}
    return {"ok": True, "customer": customer}


def get_order_status(order_id: int) -> dict:
    row = db.execute(
        "SELECT order_id, status FROM orders WHERE order_id = ?", (order_id,)
    ).fetchone()
    if row is None:
        return {"ok": False, "error": f"Order {order_id} not found"}
    return {"ok": True, "order": {"order_id": row[0], "status": row[1]}}


TOOLS = {"get_customer": get_customer, "get_order_status": get_order_status}


def run_tool(name: str, args: dict) -> dict:
    """Safely execute a tool: unknown name, bad args or errors never crash."""
    func = TOOLS.get(name)
    if func is None:
        return {"ok": False, "error": f"Unknown tool '{name}'"}
    try:
        (value,) = args.values()          # each tool takes exactly one id
        return func(int(value))
    except Exception as exc:              # bad/missing argument etc.
        return {"ok": False, "error": f"Invalid arguments for {name}: {exc}"}


def _id_param(description: str) -> types.Schema:
    return types.Schema(
        type="OBJECT",
        properties={"id": types.Schema(type="INTEGER", description=description)},
        required=["id"],
    )


declarations = [
    types.FunctionDeclaration(
        name="get_customer",
        description="Look up a customer (name and plan) by customer id.",
        parameters=types.Schema(
            type="OBJECT",
            properties={"customer_id": types.Schema(type="INTEGER", description="Customer id")},
            required=["customer_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_order_status",
        description="Get the delivery status of an order by order id.",
        parameters=types.Schema(
            type="OBJECT",
            properties={"order_id": types.Schema(type="INTEGER", description="Order id")},
            required=["order_id"],
        ),
    ),
]

SYSTEM = (
    "You are a support assistant. Use the tools to look up facts; never guess. "
    "Call every tool you need. Final reply: ONE short sentence, in this style: "
    "'Customer 2 is on the Premium plan and order 1042 is out for delivery.' "
    "Copy plan and status words exactly as the tools return them. "
    "If a tool returns an error, say plainly that the id was not found."
)

config = types.GenerateContentConfig(
    system_instruction=SYSTEM,
    tools=[types.Tool(function_declarations=declarations)],
    temperature=0,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


# ---------------------------------------------------------------- agent loop
def ask(question: str) -> None:
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]
    log = []
    answer = "No answer produced."

    for _ in range(5):  # hard cap on tool rounds
        resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
        calls = resp.function_calls
        if not calls:
            answer = (resp.text or "").strip()
            break

        contents.append(resp.candidates[0].content)
        response_parts = []
        for call in calls:
            args = dict(call.args or {})
            shown = ", ".join(str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
                              for v in args.values())
            log.append(f"Tool: {call.name}({shown})")
            result = run_tool(call.name, args)
            response_parts.append(
                types.Part.from_function_response(name=call.name, response={"result": result})
            )
        contents.append(types.Content(role="user", parts=response_parts))

    print(" | ".join(log))
    print(f"Answer: {answer}")


if __name__ == "__main__":
    print("=== Main test ===")
    ask("Which plan is customer 2 on and where is order 1042?")

    print("\n=== Unknown id ===")
    ask("Which plan is customer 99 on and where is order 5555?")

    print("\n=== Non-existent tool name (called directly) ===")
    print(run_tool("delete_customer", {"customer_id": 2}))