from rwkvstic.load import RWKV
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

def process_image_text_request(image_url, text_input):
    image = Image.open(requests.get(image_url, stream=True).raw)
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": text_input}
    ]}]
from rwkvstic.agnostic.backends import TORCH
from llama_cpp import Llama

import torch
import time
import secrets
import string
import json
from flask import Flask, Response, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

useGPU = False # True if you have a GPU.

model_mapping = {
    "rwkv-raven-7b-v8-eng-more": "models/rwkv/RWKV-4-Raven-7B-v8-EngAndMore-20230408-ctx4096.pth",
    "rwkv-raven-14b-v8-eng-more": "models/rwkv/RWKV-4-Raven-14B-v8-EngAndMore-20230408-ctx4096.pth",
    "rwkv-raven-7b-v9-eng-chn-jpn-kor": "RWKV-4-Raven-7B-v9-Eng86%25-Chn10%25-JpnEspKor2%25-Other2%25-20230414-ctx4096.pth",
    "rwkv-raven-7b-v9-eng-more": "models/rwkv/RWKV-4-Raven-7B-v9-Eng99%25-Other1%25-20230412-ctx8192.pth",
    "llama2-7b-chat": "models/llama2/llama-2-7b-chat.ggmlv3.q4_K_M.bin",
    "mistral-7b-instruct": "../../models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    "zephyr-7b": "../../models/zephyr-7b-alpha.Q4_K_M.gguf"
    # Add more models as needed
}

models = {}

alphabet = string.ascii_letters + string.digits

RWKV_INSTRUCTION_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
# Instruction:
Assistant is a large language model trained by the RWKV community. Knowledge cutoff: 2022-03 Browsing: disabled
{instruction}

# Input:
{input}

# Response:
"""

RWKV_INPUT_PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# Instruction:
{instruction}

# Response:
"""

LLAMA_INSTRUCTION_PROMPT = """Instruction: {instruction}\n[INST]{input}[/INST]"""

LLAMA_INPUT_PROMPT = """[INST]{instruction}[/INST]\n"""

MISTRAL_INSTRUCTION_PROMPT = """<s>[INST] {instruction}: {input} [/INST]"""

MISTRAL_INPUT_PROMPT = """<s>[INST] {instruction} [/INST]"""

ZEPHYR_INSTRUCTION_PROMPT = """<|system|>
{instruction}</s>
<|user|>
{input}</s>
<|assistant|>"""

ZEPHYR_INPUT_PROMPT = """<|system|>
</s>
<|user|>
{instruction}</s>
<|assistant|>"""

def generate_prompt(instruction, input=None, model_name=None):
    if model_name.startswith("rwkv"):
        if input:
            return RWKV_INSTRUCTION_PROMPT.format(instruction=instruction, input=input)
        else:
            return RWKV_INPUT_PROMPT.format(instruction=instruction)
    elif model_name.startswith("mistral"):
        if input:
            return MISTRAL_INSTRUCTION_PROMPT.format(instruction=instruction, input=input)
        else:
            return MISTRAL_INPUT_PROMPT.format(instruction=instruction)
    elif model_name.startswith("zephyr"):
        if input:
            return ZEPHYR_INSTRUCTION_PROMPT.format(instruction=instruction, input=input)
        else:
            return ZEPHYR_INPUT_PROMPT.format(instruction=instruction)
    else:
        if input:
            s = LLAMA_INSTRUCTION_PROMPT.format(instruction=instruction, input=input)
            print(s)
            return s
        else:
            s = LLAMA_INPUT_PROMPT.format(instruction=instruction)
            print(s)
            return s


def get_model(model_name):
    global models

    if not model_name in models:
        if model_name.startswith("llama"):
            model = Llama(model_path=model_mapping[model_name], n_ctx=2048)
            models[model_name] = model
            return model

        elif model_name.startswith("rwkv"):
            model_path = model_mapping.get(model_name)
            model = RWKV(model_path, mode=TORCH, useGPU=useGPU, dtype=torch.bfloat16)
            models[model_name] = model
            return model

        elif model_name.startswith("mistral"):
            model = Llama(model_path=model_mapping[model_name], n_ctx=4096, n_gpu_layers=35)
            models[model_name] = model
            return model
        
        elif model_name.startswith("zephyr"):
            model = Llama(model_path=model_mapping[model_name], n_ctx=4096, n_gpu_layers=35)
            models[model_name] = model
            return model


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


def stream_answer(model_name, system_input, user_input, max_tokens=2048):
    global models
    chat_id = f"chatcmpl-{make_id()}"
    created = int(time.time())

    stopStrings = ["# Instruction:", "# Response:", "<|endoftext|>"]
    stopTokens = [0]
    temp = 1
    top_p = 0.7

    print(f'data: {make_assistant_response(created, chat_id, model_name)}\n\n')
    yield f'data: {make_assistant_response(created, chat_id, model_name)}\n\n'

    if not model_name in models:
        model = get_model(model_name)
    else:
        model = models[model_name]

    def progressLambda(properties):
        print("progress:", properties["progress"] / properties["total"])

    if model_name.startswith("rwkv"):
        emptyState = model.emptyState
        model.setState(emptyState)

        if system_input and user_input:
            model.loadContext(newctx=generate_prompt(system_input, user_input, model_name=model_name))
        else:
            model.loadContext(newctx=generate_prompt(user_input, model_name=model_name))

    if model_name.startswith("rwkv"):
        i = 0
        while i < max_tokens:
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
    else:
        if system_input and user_input:
            stream = model.create_completion(
                generate_prompt(system_input, user_input, model_name=model_name),
                stream=True,
                max_tokens=max_tokens,
                stop=stopStrings,
                echo=True)
        elif system_input:
            stream = model.create_completion(
                generate_prompt(system_input, model_name=model_name),
                stream=True,
                max_tokens=max_tokens,
                stop=stopStrings,
                echo=True)
        else:
            stream = model.create_completion(
                generate_prompt(user_input, model_name=model_name),
                stream=True,
                max_tokens=max_tokens,
                stop=stopStrings,
                echo=True)

        result = ""
        for output in stream:
            ret = make_content_response(created, chat_id, model_name, output['choices'][0]['text'])
            print(f'data: {ret}\n\n')
            yield f'data: {ret}\n\n'

    print(f'data: {make_finish_response(created, chat_id, model_name)}\n\n')
    yield f'data: {make_finish_response(created, chat_id, model_name)}\n\n'


def generate_answer(model_name, system_input, user_input):
    global models

    print(f"model_name: {model_name}")
    print(f"system_input: {system_input}")
    print(f"user_input: {user_input}")

    if not model_name in models:
        model = get_model(model_name)
    else:
        model = models[model_name]

    number = 1024
    stopStrings = ["# Instruction:", "<|endoftext|>", "</s>"]
    stopTokens = [0]
    temp = 1
    top_p = 0.7


    def progressLambda(properties):
        print("progress:", properties["progress"] / properties["total"])

    if model_name.startswith("rwkv"):
        emptyState = model.emptyState
        model.setState(emptyState)
        if system_input and user_input:
            model.loadContext(newctx=generate_prompt(system_input, user_input, model_name=model_name))
        else:
            model.loadContext(newctx=generate_prompt(user_input, model_name=model_name))

        output = model.forward(number=number, stopStrings=stopStrings, stopTokens=stopTokens, temp=temp, top_p_usual=top_p, progressLambda=progressLambda)
        return output["output"]
    else:
        if system_input and user_input:
            output = model(generate_prompt(system_input, user_input, model_name=model_name), max_tokens=number, stop=stopStrings, echo=True)
        elif system_input:
            output = model(generate_prompt(system_input, model_name=model_name), max_tokens=number, stop=stopStrings, echo=True)
        else:
            output = model(generate_prompt(user_input, model_name=model_name), max_tokens=number, stop=stopStrings, echo=True)
        ret = output['choices'][0]['text']
        ret = ret.rpartition("[/INST]")[-1]
        return ret


@app.route('/chat/completions', methods=['POST'])
def completions():
    data = request.get_json()
    model_name = data.get('model', "mistral-7b-instruct")
    if model_name.startswith('gpt'):
        model_name = "mistral-7b-instruct"
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
