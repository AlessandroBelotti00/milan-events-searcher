# search.py
from openai import OpenAI
import time
from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI()

async def web_search(user_query: str) -> dict:
    print("Web search: Starting...")
    start = time.time()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=f"Search the latest information about: {user_query}",
        max_output_tokens=400,
        # 👇 enables web search
        tools=[{"type": "web_search"}]
    )

    result = response.output_text
    end = time.time()

    return {
        "source": "web_search",
        "status": "completed",
        "content": result,
        "latency_seconds": round(end - start, 2)
    }
