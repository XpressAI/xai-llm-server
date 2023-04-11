from flask import Flask, render_template, request

app = Flask(__name__)

from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import TORCH
import torch

useGPU = False # False
path = ""


model = RWKV(path, mode=TORCH, useGPU=useGPU, dtype=torch.bfloat16)

@app.route('/test', methods=['POST'])
def home():
    model.loadContext(newctx=f"Q: who is Jim Butcher?\n\nA:")
    output = model.forward(number=100)["output"]
    return output

if __name__ == '__main__':
    app.run(debug=True)
