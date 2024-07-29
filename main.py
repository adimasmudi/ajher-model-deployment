from flask import Flask, request, render_template, redirect, url_for,jsonify
from dotenv import load_dotenv
import os
import time

from model import BERTCorrection

load_dotenv()

# initialize our Flask application and the Keras model
app = Flask(__name__)
model = BERTCorrection()
model.load_model(os.getenv("model"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result')
def result():
    score = float(request.args.get('score'))
    reference_answer = request.args.get('reference_answer')
    answer = request.args.get('answer')
    time_cost = request.args.get('time_cost')

    # Determine score class
    if score >= 0.75:
        score_class = 'score-high'
    elif 0.5 <= score < 0.75:
        score_class = 'score-medium'
    else:
        score_class = 'score-low'

    score = 0 if score < 0 else score

    return render_template('result.html', score=score, reference_answer=reference_answer, answer=answer, time_cost=time_cost, score_class=score_class)

@app.route("/api/v1/predict", methods=["POST"])
def predict():
    response = {
        "message": "prediction success",
        "code": 200,
        "status": "success",
        "data": {}
    }
    if request.method == "POST":
        try:
            referenceAnswer = ''
            answer = ''
            if request.is_json:
                referenceAnswer = request.json.get("reference_answer")
                answer = request.json.get("answer")
            else:
                referenceAnswer = request.form.get("reference_answer")
                answer = request.form.get("answer")
            
            t1 = time.time()
            predictionResult = model.predict(referenceAnswer, answer)
            time_cost = time.time() - t1

            response["data"] = {
                "score": predictionResult[0][0],
                "time_cost": f"{time_cost} seconds"
            }

            # Redirect to result page with arguments
            return redirect(url_for('result', 
                                    score=predictionResult[0][0], 
                                    reference_answer=referenceAnswer, 
                                    answer=answer, 
                                    time_cost=f"{time_cost} seconds"))

        except Exception as err:
            response["status"] = "error"
            response["message"] = str(err)
            response["code"] = 500

    return jsonify(response) if request.is_json else redirect(url_for('result'))

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run(port=8000)