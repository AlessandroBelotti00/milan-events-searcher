import pickle
from transformers import AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm


# --------- Chunking ---------
def chunk_markdown(text: str, model_name="nomic-ai/nomic-embed-text-v1.5", token_limit=1024, stride=100):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    for i in range(0, len(input_ids), token_limit - stride):
        chunk_ids = input_ids[i:i + token_limit]
        chunk_text = tokenizer.decode(chunk_ids)
        chunks.append(chunk_text)

    print(f"Total chunks created: {len(chunks)}")
    return chunks


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

    def generate_embedding(self, contexts):
        return self.embed_model.get_text_embedding_batch(contexts)

    def embed(self, contexts):
        self.contexts = contexts
        for batch_context in tqdm(batch_iterate(contexts, self.batch_size),
                                  total=(len(contexts) + self.batch_size - 1) // self.batch_size,
                                  desc="Embedding data in batches"):
            batch_embeddings = self.generate_embedding(batch_context)
            self.embeddings.extend(batch_embeddings)


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
