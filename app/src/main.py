import uvicorn
from fastapi import FastAPI

from app.src.api.routes.chat import router as chat_router

app = FastAPI(title="Influx Agent", version="0.1.0")

app.include_router(chat_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app.src.main:app", host="0.0.0.0", port=8000, reload=True)
