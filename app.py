from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import TORCH
import torch
import os

useGPU = False # False

model_path = os.environ['model_path']
model = RWKV(model_path, mode=TORCH, useGPU=useGPU, dtype=torch.bfloat16)

def generate_answer(user_input):
    number = 100
    stopStrings = ["\n\n"]
    stopTokens = [0]
    temp = 1

    def progressLambda(properties):
        print("progress:", properties["progress"] / properties["total"])

    model.loadContext(newctx=f"Q: {user_input}\n\nA:")
    output = model.forward(number=number, stopStrings=stopStrings, stopTokens=stopTokens, temp=temp, progressLambda=progressLambda)

    return output["output"]

@app.route('/completions', methods=['POST'])
def home():
    data = request.get_json()
    user_input = data.get('user_input', "")
    answer = generate_answer(user_input)
    return jsonify(answer=answer)
    
if __name__ == '__main__':
    app.run(debug=True)
