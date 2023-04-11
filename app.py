from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import TORCH
import torch

useGPU = False # False
path = ""

model = RWKV(path, mode=TORCH, useGPU=useGPU, dtype=torch.bfloat16)

@app.route('/completions', methods=['POST'])
def home():
    data = request.get_json()
    user_input = data.get('user_input', "")
    user_prompt = f"Q: {user_input}\n\nA:"
    model.loadContext(newctx=user_prompt)
    output = model.forward(number=100)["output"]
    return jsonify(answer=output)

if __name__ == '__main__':
    app.run(debug=True)
