import ollama


class OllamaClient:
    def __init__(self, host: str):
        self.host = host
        self._client = ollama.Client(host=host)

    def chat(self, model: str, messages: list):
        return self._client.chat(
            model=model,
            messages=messages,
        )
