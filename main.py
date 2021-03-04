from flask import Flask, request
from flask_restful import Api, Resource
from transformers  import pipeline
import json
app = Flask(__name__)
api = Api(app)

nlp = pipeline(
    'question-answering',
    model='mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es',
    tokenizer=(
        'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es',
        {"use_fast": False}
    )
)

class Predict(Resource):
    @staticmethod
    def post():
        y = request.get_json()
        answer = nlp({'question':y["question"], 'context':y["text"]})
        out = {'Prediction': answer['answer'] }
        return out, 200

api.add_resource(Predict, '/predict')
if __name__ == '__main__':
    app.run(debug=True, port='8880')