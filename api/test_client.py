import json
import requests

def test_health(url="http://127.0.0.1:8000"):
    print("\n--- Testing Health Check ---")
    resp = requests.get(f"{url}/health")
    print("Status Code:", resp.status_code)
    print("Response:", resp.json())

def test_models(url="http://127.0.0.1:8000"):
    print("\n--- Testing List Models ---")
    resp = requests.get(f"{url}/v1/models")
    print("Status Code:", resp.status_code)
    print("Response:", resp.json())

def test_chat_non_streaming(url="http://127.0.0.1:8000"):
    print("\n--- Testing Chat (Non-Streaming) ---")
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "ultra-gpt-notebook",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain what SwiGLU is."}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False
    }
    
    resp = requests.post(f"{url}/v1/chat/completions", headers=headers, json=payload)
    print("Status Code:", resp.status_code)
    try:
        print("Response JSON:\n", json.dumps(resp.json(), indent=2))
    except Exception as e:
        print("Raw response:", resp.text)

def test_chat_streaming(url="http://127.0.0.1:8000"):
    print("\n--- Testing Chat (Streaming) ---")
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "ultra-gpt-notebook",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "List three main features of the Transformer model."}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": True
    }
    
    resp = requests.post(f"{url}/v1/chat/completions", headers=headers, json=payload, stream=True)
    print("Status Code:", resp.status_code)
    
    for line in resp.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            if decoded_line.startswith("data: "):
                data_content = decoded_line[6:]
                if data_content == "[DONE]":
                    print("\n[Stream Complete]")
                    break
                try:
                    chunk = json.loads(data_content)
                    delta = chunk["choices"][0]["delta"]
                    if "reasoning_content" in delta and delta["reasoning_content"]:
                        print(f"\033[33m{delta['reasoning_content']}\033[0m", end="", flush=True)
                    elif "content" in delta and delta["content"]:
                        print(delta["content"], end="", flush=True)
                except Exception as e:
                    pass

if __name__ == "__main__":
    print("=========================================")
    print("UltraGPT OpenAI Compatible Server Tester")
    print("=========================================")
    print("\nMake sure your server is running before executing this client test.")
    print("Example command to run the server:")
    print("  python api/main.py\n")
    
    # You can uncomment these calls to test when server is active
    test_health()
    test_models()
    test_chat_non_streaming()
    test_chat_streaming()
