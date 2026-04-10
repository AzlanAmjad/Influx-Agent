import logging

import uvicorn
from fastapi import FastAPI

from app.src.api.routes.openai import router as openai_router
from app.src.api.routes.schema import router as schema_router
from app.src.core.config import settings

log = logging.getLogger(__name__)

app = FastAPI(title="Influx Agent", version="0.1.0")

app.include_router(openai_router, prefix="/v1")
app.include_router(schema_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    log.info(
        "Starting uvicorn  model=%s  ollama=%s  influxdb=%s:%s",
        settings.default_model,
        settings.ollama_host,
        settings.influxdb_host,
        settings.influxdb_port,
    )
    uvicorn.run("app.src.main:app", host="0.0.0.0", port=8001, reload=True)
