import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from models.schemas import ChatRequest
from fastapi.responses import StreamingResponse
from agents.causal_agent import causal_chat, clear_messages, get_messages

router = APIRouter()


def _sse_frame(event: str, data: str) -> str:
    # SSE 规范要求多行数据每一行都以 `data:` 前缀发送，
    # 否则客户端会丢失换行后的内容，导致 markdown/code fence 破损。
    lines = str(data).splitlines() or [""]
    data_block = "\n".join(f"data: {ln}" for ln in lines)
    return f"event: {event}\n{data_block}\n\n"


def _sse_message_frame(token: str) -> str:
    # 消息 token 用 JSON 字符串编码为单行传输：
    # SSE 的 data 行拼接会丢失 token 末尾换行符（"x\n" -> "x"），
    # 导致 markdown 标题/列表/代码块全部粘连在一行无法渲染。
    return _sse_frame("message", json.dumps(str(token), ensure_ascii=False))


async def _chat_sse_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    async for item in causal_chat(
        request.message,
        request.image_url,
        request.thread_id,
        stream_events=True,
    ):
        if isinstance(item, dict):
            if item.get("type") == "message":
                yield _sse_message_frame(str(item.get("data", "")))
            elif item.get("type") == "tool":
                payload = {
                    "name": item.get("name", ""),
                    "status": item.get("status", "end"),
                    "input_preview": item.get("input_preview", ""),
                    "output_preview": item.get("output_preview", ""),
                    "ts": item.get("ts"),
                }
                yield _sse_frame("tool", json.dumps(payload, ensure_ascii=False))
            else:
                yield _sse_message_frame(str(item))
        else:
            # 兼容 fallback：把原始字符串仍当消息 token 推送
            yield _sse_message_frame(str(item))


@router.post("/chat/stream")
async def chat_endpoint(request: ChatRequest):
    """因果领域助手流式对话"""
    return StreamingResponse(
        _chat_sse_stream(request),
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


