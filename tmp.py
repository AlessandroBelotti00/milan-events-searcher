import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

# 1. Load environment variables from the .env file
# This must be called BEFORE you initialize the OpenAI client.
load_dotenv()
print("Environment variables loaded from .env file.")

# Check if the key was loaded successfully
if os.getenv("OPENAI_API_KEY"):
    print("OPENAI_API_KEY found and loaded.")
else:
    print("Error: OPENAI_API_KEY not found in environment.")
    exit()

# Initialize the OpenAI client. 
# It automatically reads os.environ["OPENAI_API_KEY"].
client = OpenAI()

def make_mock_completion_call(prompt: str) -> str:
    """
    Makes an API call to the gpt-4o-mini model.
    """
    print(f"\nSending request to gpt-4o-mini...")
    try:
        # 2. Make the API call using the retrieved key
        response: ChatCompletion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=50
        )
        
        # 3. Extract and return the response
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content
        else:
            return "Error: No response content received."

    except Exception as e:
        # This will catch errors related to an invalid key or network issues
        return f"An API error occurred: {e}"

# --- Execute the call ---
test_prompt = "Which city is better: Vicenza or Bergamo? Articulate your answer in a concise manner."
print('--- Test Prompt ---')
result = make_mock_completion_call(test_prompt)

print("\n--- Model Response ---")
print(result)
print("----------------------")