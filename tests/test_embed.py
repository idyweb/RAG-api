import os
from dotenv import load_dotenv
from google import genai
import asyncio

load_dotenv()
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

async def test():
    for model in ["text-embedding-004", "models/text-embedding-004", "gemini-embedding-001", "models/gemini-embedding-001"]:
        try:
            print(f"Testing {model}...")
            res = await client.aio.models.embed_content(model=model, contents=["test"])
            print(f"Success! {model} gave {len(res.embeddings[0].values)} dims")
            break
        except Exception as e:
            print(f"Error: {e}")

asyncio.run(test())
