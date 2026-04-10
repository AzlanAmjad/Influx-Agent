from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_host: str = "http://192.168.1.157:11434"
    default_model: str = "qwen2.5:14b"
    influxdb_host: str = "localhost"
    influxdb_port: int = 8086
    log_level: str = "DEBUG"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

