from fastapi import APIRouter
from models.schemas import ChatRequest
from fastapi.responses import StreamingResponse
from agents.causal_agent import causal_chat, clear_messages, get_messages

router = APIRouter()

@router.post("/chat/stream")
async def chat_endpoint(request: ChatRequest):
    """因果领域助手流式对话"""
    return StreamingResponse(
        causal_chat(request.message, request.image_url, request.thread_id),
        media_type="text/event-stream"
    )

@router.delete("/chat/messages")
async def clear_chat_messages(thread_id: str):
    """清空历史消息"""
    
    clear_messages(thread_id)
    return {"success":True,"message":"历史消息已清空"}

@router.get("/chat/messages")
async def get_chat_messages(thread_id: str):
    """获取历史消息"""
    messages = get_messages(thread_id)
    return {"messages":messages}


