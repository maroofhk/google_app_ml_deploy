from flask import Flask, request
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# http://localhost:5000/api_predict
with open("flower_v1.pkl", "rb") as f:
    model_pk = pickle.load(f)


@app.route('/api_predict', methods=['POST', 'GET'])
def api_predict():
    if request.method == 'GET':
        return "Please Send POST Request"
    elif request.method == 'POST':

        print("Hello" + str(request.get_json()))

        data = request.get_json()

        sepal_length = data["sepal_length"]
        sepal_width = data["sepal_width"]
        petal_length = data["petal_length"]
        petal_width = data["petal_width"]

        data = np.array([[sepal_length, sepal_width,
                          petal_length, petal_width]])

        prediction = model_pk.predict(data)
        return str(prediction)


if __name__ == "__main__":
    app.run()