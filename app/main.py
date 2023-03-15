import numpy as np
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

from fastapi import FastAPI

app = FastAPI()

X = np.array([-1, -2, 2, 1, 2, 1]).reshape(-1, 1)
y = np.array([1, 1, 0, 0, 1, 0])
lr = LogisticRegression()
lr.fit(X, y)
score = lr.score(X, y)
# print("Score: %s" % score)
mlflow.log_metric("score", score)
mlflow.sklearn.log_model(lr, "model")
# print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

@app.get("/")
def read_root():
    return {"Score: %s": f"{score}", "Model saved in run %s":f"{mlflow.active_run().info.run_uuid}"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}