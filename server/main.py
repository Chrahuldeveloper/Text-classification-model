from sentence_transformers import SentenceTransformer
import numpy as np
from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str


model = SentenceTransformer("all-MiniLM-L6-v2")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    exp_scores = np.exp(z)
    return exp_scores / np.sum(exp_scores)


# Binary Classification

weights = np.load("weights.npy")
bias = np.load("bias.npy")[0]


def get_useful_result(request: TextRequest):

    text = request.text

    embedding = model.encode([text])[0]

    z = embedding @ weights + bias

    probability = sigmoid(z)

    print("Probability:", probability)

    return {
        "probability": float(probability),
        "useful": bool(probability >= 0.5)
    }


# Multiclass Classification

weights1 = np.load("weights1.npy")
bias1 = np.load("bias1.npy")


def get_category_result(request: TextRequest):

    text = request.text

    embedding = model.encode([text])[0]

    z = embedding @ weights1 + bias1

    probability = softmax(z)

    probabilities = probability.tolist()[0]

    categories = ["Sports", "Science", "Politics"]

    return {
        "sports": probabilities[0],
        "science": probabilities[1],
        "politics": probabilities[2],
        "ans": categories[int(np.argmax(probabilities))]
    }