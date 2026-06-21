from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

class ChatCompletionResponseStreamDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: ChatCompletionResponseStreamDelta
    finish_reason: Optional[str] = None

class ChatCompletionResponseChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionResponseStreamChoice]

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = 1677610602
    owned_by: str = "organization-owner"

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelObject]

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 16
    stop: Optional[Union[str, List[str]]] = None

class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[Dict] = None
    finish_reason: Optional[str] = "stop"

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo

class CompletionResponseChunkChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[Dict] = None
    finish_reason: Optional[str] = None

class CompletionResponseChunk(BaseModel):
    id: str
    object: str = "text_completion.chunk"
    created: int
    model: str
    choices: List[CompletionResponseChunkChoice]
