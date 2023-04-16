from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import TORCH
import torch
import os
import time
import secrets
import string
import json
from flask import Flask, Response, render_template, request, jsonify

app = Flask(__name__)

useGPU = False # False

model_mapping = {
    "rwkv-raven-7b-v8-eng-more": "models/rwkv/RWKV-4-Raven-7B-v8-EngAndMore-20230408-ctx4096.pth",
    "rwkv-raven-14b-v8-eng-more": "models/rwkv/RWKV-4-Raven-14B-v8-EngAndMore-20230408-ctx4096.pth",
    "rwkv-raven-7b-v9-eng-chn-jpn-kor": "RWKV-4-Raven-7B-v9-Eng86%25-Chn10%25-JpnEspKor2%25-Other2%25-20230414-ctx4096.pth",
    "rwkv-raven-7b-v9-eng-more": "models/rwkv/RWKV-4-Raven-7B-v9-Eng99%25-Other1%25-20230412-ctx8192.pth"
    # Add more models as needed
}

models = {}

alphabet = string.ascii_letters + string.digits

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
# Instruction:
Assistant is a large language model trained by the RWKV community. Knowledge cutoff: 2022-03 Browsing: disabled
{instruction}

# Input:
{input}

# Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
# Instruction:
{instruction}

# Response:
"""

def make_assistant_response(created, chat_id, model_name):
    return json.dumps({
        "choices": [
            {
                "delta": {
                    "role": "assistant"
                },
                "finish_reason": None,
                "index": 0
            }
        ],
        "created": created,
        "id": chat_id,
        "model": model_name,
        "object": "chat.completion.chunk"
    })

def make_content_response(created, chat_id, model_name, content):
    return json.dumps({
            "choices": [
                {
                    "delta": {
                        "content": content
                    },
                    "finish_reason": None,
                    "index": 0
                }
            ],
            "created": created,
            "id": chat_id,
            "model": model_name,
            "object": "chat.completion.chunk"
        })

def make_finish_response(created, chat_id, model_name):
    return json.dumps({
        "choices": [
            {
                "delta": {},
                "finish_reason": "stop",
                "index": 0
            }
        ],
        "created": created,
        "id": chat_id,
        "model": model_name,
        "object": "chat.completion.chunk"
    })

def make_id():
    return ''.join(secrets.choice(alphabet) for i in range(29))

def stream_answer(model_name, system_input, user_input):
    global models
    chat_id = f"chatcmpl-{make_id()}"
    created = int(time.time())
    number = 100
    stopStrings = ["# Instruction:", "# Response:", "<|endoftext|>"]
    stopTokens = [0]
    temp = 1
    top_p = 0.7

    print(f'data: {make_assistant_response(created, chat_id, model_name)}\n\n')
    yield f'data: {make_assistant_response(created, chat_id, model_name)}\n\n'

    if not model_name in models:
        model_path = model_mapping.get(model_name)
        model = RWKV(model_path, mode=TORCH, useGPU=useGPU, dtype=torch.float32)
    else:
        model = models[model_name]
        
    def progressLambda(properties):
        print("progress:", properties["progress"] / properties["total"])

    emptyState = model.emptyState
    model.setState(emptyState)
    if system_input and user_input:
        model.loadContext(newctx=generate_prompt(system_input, user_input))
    else:
        model.loadContext(newctx=generate_prompt(user_input))
    i = 0
    while i < number:
        output = model.forward(number=5, stopStrings=stopStrings, stopTokens=stopTokens, temp=temp, top_p_usual=top_p, progressLambda=progressLambda)
        i += 5
        for stopString in stopStrings:
            if stopString in output["output"]:
                ret = make_content_response(created, chat_id, model_name, output["output"].replace(stopString, ""))
                print(f'data: {ret}\n\n')
                yield f'data: {ret}\n\n'
                print(f'data: {make_finish_response(created, chat_id, model_name)}\n\n')
                yield f'data: {make_finish_response(created, chat_id, model_name)}\n\n'
                return

        ret = make_content_response(created, chat_id, model_name, output["output"])
        print(f'data: {ret}\n\n')
        yield f'data: {ret}\n\n'

    print(f'data: {make_finish_response(created, chat_id, model_name)}\n\n')
    yield f'data: {make_finish_response(created, chat_id, model_name)}\n\n'
                

def generate_answer(model_name, system_input, user_input):
    global models

    if not model_name in models:
        model_path = model_mapping.get(model_name)
        model = RWKV(model_path, mode=TORCH, useGPU=useGPU, dtype=torch.bfloat16)
        models[model_name] = model
    else:
        model = models[model_name]

    number = 100
    stopStrings = ["# Instruction:", "<|endoftext|>"]
    stopTokens = [0]
    temp = 1
    top_p = 0.7


    def progressLambda(properties):
        print("progress:", properties["progress"] / properties["total"])

    emptyState = model.emptyState
    model.setState(emptyState)
    if system_input and user_input:
        model.loadContext(newctx=generate_prompt(system_input, user_input))
    else:
        model.loadContext(newctx=generate_prompt(user_input))
    output = model.forward(number=number, stopStrings=stopStrings, stopTokens=stopTokens, temp=temp, top_p_usual=top_p, progressLambda=progressLambda)

    return output["output"]

@app.route('/v1/chat/completions', methods=['POST'])
def completions():
    data = request.get_json()
    model_name = data.get('model', "")
    messages = data.get('messages', [])
    stream = data.get('stream', False)
    chat_id = f"chatcmpl-{make_id()}"
    created = int(time.time())
    
    if not messages:
        return jsonify(error="No messages provided"), 400

    user_input = "\n".join([m['content'] for m in messages if m['role'] != 'system'])
    system_input = "\n".join([m['content'] for m in messages if m['role'] == 'system'])

    if not stream:
        answer = generate_answer(model_name, system_input, user_input)
        
        if not answer:
            return jsonify(error=f"Model '{model_name}' not found"), 404

        return jsonify(
            {
                "id": chat_id,
                "object": "chat.completion",
                "created": created,
                "choices": [{
                    "index": 0,
                    "message": {
                    "role": "assistant",
                    "content": answer,
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
        )
    else:
        return Response(stream_answer(model_name, system_input, user_input), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
