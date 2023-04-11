# xai-llm-server

## Setup
1. Run setup
    ```
    $ bash setup.sh
    ```
2. Update the model path in app.py
3. Run the server using `python app.py`.

## Sending Requests
```
curl http://localhost:5000/completions  -H "Content-Type: application/json"  -d '{
    "model": "model1",
    "messages": [{"role": "user", "content": "What is the capital of Indonesia?"}]
  }'
```