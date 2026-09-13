from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            
    allow_credentials=True,           
    allow_methods=["*"],             
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str


model = SentenceTransformer("all-MiniLM-L6-v2")

weights = np.load("weights.npy")
bias = np.load("bias.npy")[0]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    exp_scores = np.exp(z)
    return exp_scores / np.sum(exp_scores)

@app.post("/check-text")
def get_result(request: TextRequest):
    try:
        text = request.text
        embedding = model.encode([text])[0]
        z = embedding @ weights + bias
        probability = sigmoid(z)
        print("Probability:", probability)
        if probability >= 0.5:
            return {
                "probability" : probability,
                "useful" : True
            }
        else:
            return {
                "probability" : probability,
                "useful" : False
            }
    except Exception as e:
        print(e)        


weights1 = np.load("weights1.npy")
bias1 = np.load("bias1.npy")

@app.post("/classify-text")
def get_result(request: TextRequest):
    try:
        text = request.text
        embedding = model.encode([text])[0]
        z = embedding @ weights1 + bias1
        probability = softmax(z)
        print(probability.tolist()[0])
        return {
            "sports": probability.tolist()[0][0],
            "science": probability.tolist()[0][1],
            "politics": probability.tolist()[0][2],
            "ans" : int(np.argmax(probability.tolist()[0]))
        }
    except Exception as e:
        print(e)        





