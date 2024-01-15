from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

from model import BERTCorrection

load_dotenv()

# initialize our Flask application and the Keras model
app = Flask(__name__)
model = BERTCorrection()
model.load_model(os.getenv("model"))

@app.route("/api/predict", methods=["POST"])
def predict():
    response = {
        "message" : "prediction success",
        "code" : 200,
        "status" : "success",
        "data" : {}
    }
    if request.method == "POST":
        try:
            if request.is_json:
                referenceAnswer = request.json.get("reference_answer")
                answer = request.json.get("answer")

                predictionResult = model.predict(referenceAnswer, answer)[0][0]
                uniqueNess = model.processUniqueness(answer)

                predictionResult = (predictionResult * 95)/100
                uniqueNess = (uniqueNess * 5) /100

                finalResult = round((predictionResult+uniqueNess),2)*100 if (predictionResult+uniqueNess)*100 < 100 else 100

                response["data"]["grade"] = finalResult
            else:
                response["status"] = "error"
                response["message"] = "request must be json"
                response["code"] = 500

        except Exception as err:
            
            response["status"] = "error"
            response["message"] = err
            response["code"] = 500

    # return the data dictionary as a JSON response
    return jsonify(response)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run(port=8000)