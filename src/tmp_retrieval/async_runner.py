import asyncio
import time
from retriever import rag_retrieval
from web_search import web_search


async def main():
    start_time = time.time()

    # Run both tasks concurrently
    tasks = {
        "rag_retrieval": asyncio.create_task(rag_retrieval("Find events in Milan")),
        "web_search": asyncio.create_task(web_search("Find events in Milan"))
    }

    # Process results as they complete
    for task_name, task in tasks.items():
        result = await task
        elapsed_time = time.time() - start_time
        print(f"[{task_name}] Completed in {elapsed_time:.2f} seconds:")
        print(result)

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())