import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError, field_validator

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = (
    "Extract support ticket details and reply with ONLY a JSON object with these keys: "
    "customer_name (string), product (string), "
    "issue_type (string, e.g. billing, technical, account, how-to), "
    "urgency (integer, how urgent the customer feels the problem is), "
    "summary (one short sentence). No markdown, no extra text."
)

TICKETS = [
    "Hi, I'm Priya Sharma. My CloudSync desktop app crashes every time I open it. "
    "Annoying, but not urgent.",
    "This is Daniel Wu. I was double charged for my Pro plan on BillPro this month. "
    "Please fix it soon.",
    "Maria Lopez here. Our whole team is locked out of TeamChat and we have a client "
    "demo in one hour. This is a 10/10 emergency, urgency 10 out of 10!!!",
    "Hello, Ahmed Khan. Could you tell me how to export my reports from DataViz? No rush.",
    "Sarah O'Neil: the invoice PDF in BillPro shows a blank page. Need it by end of week.",
]


class TicketInfo(BaseModel):
    customer_name: str
    product: str
    issue_type: str
    urgency: int
    summary: str

    @field_validator("urgency")
    @classmethod
    def check_urgency(cls, v):
        if not 1 <= v <= 5:
            raise ValueError("urgency must be 1-5")
        return v


def short_error(e: ValidationError) -> str:
    msgs = [err["msg"].replace("Value error, ", "") for err in e.errors()]
    return "; ".join(msgs)


def call_model(contents):
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0,
                    response_mime_type="application/json",  # JSON mode
                ),
            )
            return response.text or ""
        except errors.ClientError as e:
            if getattr(e, "code", None) == 429:
                wait = 20 * (attempt + 1)
                print(f"  rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
    return ""


def extract_ticket(text: str) -> dict:
    contents = [types.Content(role="user", parts=[types.Part(text=f"Ticket: {text}")])]
    last_error = ""
    for n in range(1, MAX_ATTEMPTS + 1):
        raw = call_model(contents)
        time.sleep(4)  # stay under rate limit
        try:
            info = TicketInfo.model_validate_json(raw)
            print(f"Attempt {n}: success")
            return {"success": True, "data": info.model_dump(), "attempts": n, "error": None}
        except ValidationError as e:
            last_error = short_error(e)
            print(f"Attempt {n}: validation error ({last_error})")
            contents.append(types.Content(role="model", parts=[types.Part(text=raw)]))
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=(
                        f"Your previous response failed validation: {last_error}. "
                        "Return corrected JSON only, with urgency as an integer from 1 to 5."
                    ))],
                )
            )
    return {
        "success": False,
        "data": None,
        "attempts": MAX_ATTEMPTS,
        "error": f"Failed after {MAX_ATTEMPTS} attempts: {last_error}",
    }


if __name__ == "__main__":
    # python extract.py      -> all 5 tickets
    # python extract.py 3    -> only ticket 3 (clean demo of the retry)
    picks = [int(a) for a in sys.argv[1:]] or list(range(1, len(TICKETS) + 1))
    results = {}
    for i in picks:
        if len(picks) > 1:
            print(f"\nTicket {i}:")
        results[i] = extract_ticket(TICKETS[i - 1])
    print("\nResults:")
    for i, r in results.items():
        print(f"Ticket {i}: {r['data'] if r['success'] else 'FAILED - ' + r['error']}")