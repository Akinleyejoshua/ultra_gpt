# UltraGPT FastAPI Server

An industry-standard, production-ready, OpenAI-compatible FastAPI server for the UltraGPT model. This allows you to query your custom UltraGPT model using the official `openai` SDK or direct HTTP requests.

## Features

- **OpenAI-Compatible Chat Completions**: Implements both `/v1/chat/completions` (supporting Server-Sent Events (SSE) streaming) and `/v1/models`.
- **FastAPI Lifespan Integration**: Loads the model and weights once at startup to keep memory footprints low and avoid loading latency during requests.
- **Serialized Model Execution**: Uses `asyncio.Lock` and background threads to prevent TensorFlow execution race conditions and blockages in the async event loop.
- **Context Window Pruning**: Automatically formats chat history into the ChatML template format and prunes older turns if history exceeds the context budget.

## Setup & Running the Server

1. **Activate the Environment**:
   ```bash
   conda activate ml_env
   ```

2. **Start the Server**:
   You can start the server directly using defaults (which automatically loads the notebook configuration and latest trained weights):
   ```bash
   python api/main.py --host 127.0.0.1 --port 8000
   ```

## Endpoints

- `GET /health`: Health-check probe indicating server status and loaded model.
- `GET /v1/models`: Returns details of the active model.
- `POST /v1/chat/completions`: Generates response completions.

### Example Request Payloads

#### Non-Streaming JSON Response
```json
{
  "model": "ultra-gpt-small",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 150,
  "stream": false
}
```

#### Streaming Responses (Server-Sent Events)
Setting `"stream": true` will yield standard SSE events matching the OpenAI completion chunks structure:
```json
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
```

## Client Testing

Run the included test client to query your server:
```bash
python api/test_client.py
```

### Querying with OpenAI python library

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="ultra-gpt-small",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain attention mechanism."}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```
