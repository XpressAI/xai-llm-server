from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import TORCH
import torch
import os

useGPU = False # False

model_mapping = {
    "model1": "/path/to/model1.pth",
    "model2": "path/to/model2.pth",
    # Add more models as needed
}

def generate_answer(model_name, user_input):

    model_path = model_mapping.get(model_name)

    if not model_path:
        return None

    model = RWKV(model_path, mode=TORCH, useGPU=useGPU, dtype=torch.bfloat16)

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
def completions():
    data = request.get_json()
    model_name = data.get('model', "")
    messages = data.get('messages', [])
    
    if not messages:
        return jsonify(error="No messages provided"), 400

    user_input = messages[-1].get('content', "")
    answer = generate_answer(model_name, user_input)
    
    if not answer:
        return jsonify(error=f"Model '{model_name}' not found"), 404

    return jsonify(answer=answer)

if __name__ == '__main__':
    app.run(debug=True)