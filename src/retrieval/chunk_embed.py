import pickle
from transformers import AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm
import os
import numpy as np
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv(override=True)

# --------- Chunking ---------

class Chunk(BaseModel):
    """A food recipe chunk with structured fields."""
    title: str = Field(..., description="The title of the recipe")
    ingredients: str = Field(..., description="The list of ingredients needed for the recipe and their quantities")
    preparation: str = Field(..., description="The preparation steps for the recipe")
    cooking_mode: str = Field(..., description="The list of cooking modes to be used (e.g., baking, frying, boiling, oven)")

class ChunkList(BaseModel):
    chunks: list[Chunk]

def chunk_to_embedding_text(chunk: ChunkList) -> str:
    return f"""
        Recipe title: {chunk.title}

        Ingredients:
        {chunk.ingredients}

        Preparation:
        {chunk.preparation}

        Cooking mode:
        {chunk.cooking_mode}
    """.strip()

def chunking_llm(paragraph):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI(model=os.getenv("OPENAI_DEPLOYMENT_NAME"))

    model_with_structure = model.with_structured_output(ChunkList)

    prompt = f"""
        You are extracting structured recipe data from a document.

        Split the text into multiple chunks.
        Each chunk MUST correspond to exactly ONE recipe.

        Rules:
        - Do not merge recipes
        - Do not invent information
        - If a field is missing, return an empty string
        - Return ONLY structured data

        Text:
        {paragraph}
        """

    response = model_with_structure.invoke(prompt)
    return response


# --------- Embedding ---------
def batch_iterate(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i: i + batch_size]


class EmbedData:
    def __init__(self, embed_model_name="nomic-ai/nomic-embed-text-v1.5", batch_size=8):
        self.embed_model_name = embed_model_name
        self.embed_model = self._load_embed_model()
        self.batch_size = batch_size
        self.embeddings = []
        self.contexts = []

    def _load_embed_model(self):
        return HuggingFaceEmbedding(model_name=self.embed_model_name,
                                    trust_remote_code=True,
                                    cache_folder='./hf_cache')

    def generate_embedding(self, chunk: Chunk) -> np.ndarray:
   
        # for chunk in [chunk_to_embedding_text(chunk) for chunk in chunk_list.chunks]:  # iterate over each Chunk
        # Embed each field separately
        title_emb = np.array(
            self.embed_model.get_text_embedding(chunk.title),
            dtype=np.float32
        ).flatten()

        ingredients_emb = np.array(
            self.embed_model.get_text_embedding(chunk.ingredients),
            dtype=np.float32
        ).flatten()

        prep_emb = np.array(
            self.embed_model.get_text_embedding(chunk.preparation),
            dtype=np.float32
        ).flatten()

        mode_emb = np.array(
            self.embed_model.get_text_embedding(chunk.cooking_mode),
            dtype=np.float32
        ).flatten()

        # Weighted average (adjust weights as needed)
        final_emb = (
            0.5 * ingredients_emb +
            0.3 * prep_emb +
            0.2 * title_emb +
            0.0 * mode_emb  # optional
        )
        
        return final_emb

    def embed(self, contexts: ChunkList):

        for el in contexts.chunks:
            embeddings = self.generate_embedding(el)
            self.embeddings.append(embeddings)


# --------- Save / Load ---------
def save_embeddings(embeddata, filename):
    data = {
        "contexts": embeddata.contexts,
        "embeddings": embeddata.embeddings
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Embeddings saved to {filename}")


def load_embeddings(filename, embed_model_name="nomic-ai/nomic-embed-text-v1.5", batch_size=8):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    embeddata = EmbedData(embed_model_name=embed_model_name, batch_size=batch_size)
    embeddata.contexts = data["contexts"]
    embeddata.embeddings = data["embeddings"]
    return embeddata
