from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_host: str = "http://192.168.1.157:11434"
    default_model: str = "gemma4:26b-a4b-it-q8_0"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
