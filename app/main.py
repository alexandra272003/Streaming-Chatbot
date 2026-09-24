from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.errors import AppError, app_error_handler
from app.routers import conversations, stream, ws

app = FastAPI(title="Streaming Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(AppError, app_error_handler)

app.include_router(conversations.router)
app.include_router(stream.router)
app.include_router(ws.router)


@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "pong"}
