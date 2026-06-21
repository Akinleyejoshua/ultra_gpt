import os
import sys
import json

# Disable HDF5 file locking to prevent BlockingIOError
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# import tensorflow as tf
# tf.get_logger().setLevel("ERROR")

import uuid
import time
import argparse
import asyncio
import queue
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
    ChatCompletionResponseChunk,
    ChatCompletionResponseStreamChoice,
    ChatCompletionResponseStreamDelta,
    ModelListResponse,
    ModelObject,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseChunk,
    CompletionResponseChunkChoice,
)
from api.server import ModelServer

model_server: ModelServer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model server on startup
    global model_server
    preset = app.state.preset
    weights = app.state.weights
    model_server = ModelServer(preset=preset, weights_path=weights)
    yield
    # Shutdown logic if needed
    print("[FastAPI] Shutting down LLM server.")

app = FastAPI(
    title="UltraGPT OpenAI Compatible Server",
    description="FastAPI server implementing OpenAI compatible chat completion endpoints for UltraGPT.",
    version="1.0.0",
    lifespan=lifespan
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.state.preset = "notebook"
app.state.weights = os.path.join(project_root, "output", "notebook_checkpoints", "ultra_gpt_notebook_latest.weights.h5")

# Enable CORS for generic API consumption
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Simple status/health probe endpoint."""
    return {"status": "ok", "model": app.state.preset, "timestamp": time.time()}

@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """Lists the current active model in OpenAI convention."""
    model_id = f"ultra-gpt-{app.state.preset}"
    return ModelListResponse(data=[ModelObject(id=model_id, created=int(time.time()))])

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI compatible Chat Completion endpoint."""
    if not model_server:
        raise HTTPException(status_code=503, detail="Model server is not initialized.")
    
    chat_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    
    # Prune and format the incoming message history
    max_tokens = req.max_tokens or 150
    prompt, actual_max_tokens = model_server.format_and_prune_prompt(req.messages, max_tokens)
    
    if req.stream:
        async def sse_generator():
            # Acquire lock to serialize model inference
            async with model_server.lock:
                q = queue.Queue()
                # Run the generation inside a separate thread to prevent blocking
                thread = threading.Thread(
                    target=model_server.generate_stream,
                    args=(prompt, actual_max_tokens, req.temperature, req.top_k, req.top_p, q)
                )
                thread.start()
                
                # 1. Send the assistant role delta
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                
                # 2. Yield token deltas as they arrive
                while True:
                    try:
                        item = await asyncio.to_thread(q.get, timeout=15.0)
                    except queue.Empty:
                        break
                    
                    if item is None:  # End of stream sentinel
                        chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": req.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        break
                        
                    if isinstance(item, Exception):
                        import traceback
                        print("[ModelServer Error during streaming]:", file=sys.stderr)
                        traceback.print_exception(type(item), item, item.__traceback__, file=sys.stderr)
                        break
                    
                    if item:
                        chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": req.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": item},
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        
                yield "data: [DONE]\n\n"
                
        return StreamingResponse(sse_generator(), media_type="text/event-stream")
        
    else:
        # Standard non-streaming POST request
        async with model_server.lock:
            response_text = await asyncio.to_thread(
                model_server.generate_sync,
                prompt,
                actual_max_tokens,
                req.temperature,
                req.top_k,
                req.top_p
            )
            
        prompt_tokens = len(model_server.tokenizer.encode(prompt))
        completion_tokens = len(model_server.tokenizer.encode(response_text))
        
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=response_text),
            finish_reason="stop"
        )
        
        return ChatCompletionResponse(
            id=chat_id,
            created=created_time,
            model=req.model,
            choices=[choice],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    """OpenAI compatible legacy Text Completion endpoint."""
    if not model_server:
        raise HTTPException(status_code=503, detail="Model server is not initialized.")
    
    completion_id = f"cmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    
    # Extract prompt
    if isinstance(req.prompt, list):
        prompt = " ".join(req.prompt)
    else:
        prompt = req.prompt
        
    # Prune prompt based on context length limit
    max_tokens = req.max_tokens or 150
    tokens = model_server.tokenizer.encode(prompt)
    max_prompt_len = model_server.config.block_size - max_tokens
    if len(tokens) > max_prompt_len:
        tokens = tokens[-max_prompt_len:]
        prompt = model_server.tokenizer.decode(tokens)
        
    actual_max_tokens = max_tokens
    
    if req.stream:
        async def sse_generator():
            async with model_server.lock:
                q = queue.Queue()
                thread = threading.Thread(
                    target=model_server.generate_stream,
                    args=(prompt, actual_max_tokens, req.temperature, req.top_k, req.top_p, q)
                )
                thread.start()
                
                while True:
                    try:
                        item = await asyncio.to_thread(q.get, timeout=15.0)
                    except queue.Empty:
                        break
                        
                    if item is None:
                        choice = CompletionResponseChunkChoice(
                            index=0,
                            text="",
                            finish_reason="stop"
                        )
                        chunk = CompletionResponseChunk(
                            id=completion_id,
                            created=created_time,
                            model=req.model,
                            choices=[choice]
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                        break
                        
                    if isinstance(item, Exception):
                        import traceback
                        print("[ModelServer Error during streaming completions]:", file=sys.stderr)
                        traceback.print_exception(type(item), item, item.__traceback__, file=sys.stderr)
                        break
                        
                    if item:
                        choice = CompletionResponseChunkChoice(
                            index=0,
                            text=item,
                            finish_reason=None
                        )
                        chunk = CompletionResponseChunk(
                            id=completion_id,
                            created=created_time,
                            model=req.model,
                            choices=[choice]
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                        
                yield "data: [DONE]\n\n"
                
        return StreamingResponse(sse_generator(), media_type="text/event-stream")
    else:
        async with model_server.lock:
            response_text = await asyncio.to_thread(
                model_server.generate_sync,
                prompt,
                actual_max_tokens,
                req.temperature,
                req.top_k,
                req.top_p
            )
            
        prompt_tokens = len(model_server.tokenizer.encode(prompt))
        completion_tokens = len(model_server.tokenizer.encode(response_text))
        
        choice = CompletionResponseChoice(
            index=0,
            text=response_text,
            finish_reason="stop"
        )
        
        return CompletionResponse(
            id=completion_id,
            created=created_time,
            model=req.model,
            choices=[choice],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_weights = os.path.join(project_root, "output", "notebook_checkpoints", "ultra_gpt_notebook_latest.weights.h5")

    parser = argparse.ArgumentParser(description="Start UltraGPT FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--preset", choices=["toy", "small", "medium", "notebook"], default="notebook", help="Model size preset")
    parser.add_argument("--weights", default=default_weights, help="Path to weights file (.weights.h5)")
    args = parser.parse_args()
    
    weights_path = args.weights
    if not os.path.isabs(weights_path):
        if not os.path.exists(weights_path):
            weights_path = os.path.join(project_root, weights_path)

    app.state.preset = args.preset
    app.state.weights = weights_path
    
    import uvicorn
    uvicorn.run("main:app", host=args.host, port=args.port, reload=True)

if __name__ == "__main__":
    main()
