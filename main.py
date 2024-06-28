import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

class Sentences(BaseModel):
    sentences: List[str]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#model = SentenceTransformer('clibrain/Llama-2-7b-ft-instruct-es-sharded-bf16')

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(sentence:Sentences):

    emb_a = model.encode(sentence.sentences[0], convert_to_tensor= False)
    emb_b = model.encode(sentence.sentences[1], convert_to_tensor= False)

    similarity = np.dot(emb_a, emb_b) / (norm(emb_a) * norm(emb_b))

    return{
            "source" : sentence.sentences[0],
            "target" : sentence.sentences[1],
            "similarity": str(similarity)
        }

    """ if __name__ == '__main__':
        sentences = ["This is an example sentence", "Each sentence is converted"]
        sentences = ["12 DE INFANT.", "R 12 INFANTERIA"]
    """

    """ embeddings = model.encode(sentences)"""