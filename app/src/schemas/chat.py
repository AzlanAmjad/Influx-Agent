from pydantic import BaseModel
from typing import List, Literal


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: List[Message]
    stream: bool = False
