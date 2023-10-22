# Popular AI company compatible LLM Server

This is a very simple Flask application that provides a popular compatible API for other large language models.

Very useful if you have tests or lots of running [Collaborative Agent Modules](https://github.com/xpressai/xai-gpt-agent-toolkit) :-)

It currently supports Llama2, Mistral-7b and RWKV since these models can run pretty easily on
local hardware which makes it a great fit for the agent use case.

Streaming is supported as well.

## Setup
1. Create a venv `python3 -m venv venv`
2. Activate venv `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
3. Install dependencies `pip install -r requirements.txt`
4. Create a symlink to your models. Example `ln -s /mnt/ssd/models/rwkv models/rwkv`
5. Run the server using `python app.py`.

## Sending Requests
```
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer WE_DONT_NEED_NO_STINKING_TOKENS" \
  -d '{
    "model": "mistral-7b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```
