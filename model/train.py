from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


model = SentenceTransformer("all-MiniLM-L6-v2")

df  = pd.read_csv("./data/data.csv")

X  = model.encode(
    df["text"].tolist(),
    show_progress_bar=True
)

Y = df["label"].to_numpy()


weights = np.zeros(X.shape[1])
bias = 0.0

def sigmoid(z):
    return 1 / (1+np.exp(-z))

lr = 0.01
epochs = 1000


for epoch in range(epochs):
    z =  X @ weights + bias
    predictions = sigmoid(z)
    dw = (X.T @ (predictions - Y)) / len(Y)
    db = np.mean(predictions - Y)

    weights -= lr * dw
    bias -= lr * db

np.save("weights.npy", weights)
np.save("bias.npy", np.array([bias]))

print("done")


# text classification
df  = pd.read_csv("./data/data1.csv")

X = model.encode(df["text"].to_list(),show_progress_bar=True)
Y = df["label"].to_numpy()
weights = np.zeros((384,3))
bias = np.zeros(3)
lr  = 0.01
epochs = 1500


def softmax(z):
    expo_scores = np.exp(z)
    return expo_scores / np.sum(expo_scores,axis=1,keepdims=True)

for epoch in range(epochs):
    z = X @ weights + bias
    predictions = softmax(z)
    error = predictions.copy()
    for i in range(len(Y)):
        error[i, Y[i]] -= 1

    dw = (X.T @ error) / len(Y)
    db = np.mean(error, axis=0)

    weights -= lr * dw
    bias -= lr * db

np.save("weights1.npy", weights)
np.save("bias1.npy", np.array([bias]))

print("done")

