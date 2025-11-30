# retriever.py
import asyncio
import random

async def rag_retrieval(user_query: str) -> dict:
    print("RAG: Starting vector search simulation...")
    await asyncio.sleep(random.uniform(3, 6))  # Simulate latency

    mock_context = """
        Event: Christmas Market in Piazza Duomo
        Dates: Early December – January 6
        Attractions: Local crafts, food stalls, Christmas tree
    """

    return {
        "source": "mock_rag",
        "status": "completed",
        "content": f"Based on knowledge base 🚂 — {mock_context.strip()}",
        "latency_seconds": "3-6"
    }
