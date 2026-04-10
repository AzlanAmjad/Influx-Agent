from app.src.llm.ollama_client import OllamaClient
from app.src.schemas.chat import Message


class AgentService:
    def __init__(self, client: OllamaClient, default_model: str):
        self.client = client
        self.default_model = default_model

    def run(self, messages: list[Message], model: str | None = None):
        model = model or self.default_model

        response = self.client.chat(
            model=model,
            messages=[m.model_dump() for m in messages],
        )

        return response
