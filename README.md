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
curl -X POST -H "Content-Type: application/json" -d '{"user_input": "What is the capital of France?"}' http://localhost:5000/completions
```