import json
import asyncio
from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session

from agents.crag_graph import app as crag_app
from database import get_db, Conversation, Message

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        })

@app.get("/api/conversations")
def get_conversations(db: Session = Depends(get_db)):
    convos = db.query(Conversation).order_by(Conversation.created_at.desc()).all()
    return [{"id": c.id, "title": c.title} for c in convos]

@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    msgs = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.created_at.asc()).all()
    if not msgs:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return [
        {
            "role": m.role,
            "content": m.content,
            "sources": json.loads(m.sources) if m.sources else None,
            "trace": json.loads(m.trace) if m.trace else None
        } for m in msgs
    ]

@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: int, db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conv)
    db.commit()
    return {"status": "success"}

@app.get("/chat")
async def chat_endpoint(request: Request, q: str, conversation_id: int = None, db: Session = Depends(get_db)):
    async def event_generator():
        nonlocal conversation_id
        
        # Load chat history
        chat_history = []
        if conversation_id:
            db_messages = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.created_at.asc()).all()
            for m in db_messages:
                chat_history.append({"role": m.role, "content": m.content})
        else:
            # Create new conversation
            new_conv = Conversation(title=q[:50])
            db.add(new_conv)
            db.commit()
            db.refresh(new_conv)
            conversation_id = new_conv.id
            
            yield {
                "event": "init",
                "data": json.dumps({"conversation_id": conversation_id})
            }

        # Save user message immediately
        user_msg = Message(conversation_id=conversation_id, role="user", content=q)
        db.add(user_msg)
        db.commit()

        # Run LangGraph
        final_answer = ""
        final_sources = []
        
        try:
            async for state_update in crag_app.astream({"question": q, "chat_history": chat_history}, stream_mode="updates"):
                if await request.is_disconnected():
                    break
                
                node_name = list(state_update.keys())[0]
                node_state = state_update[node_name]
                
                payload = {"node": node_name}
                
                if node_name == "eval_each_doc":
                    payload["verdict"] = node_state.get("verdict")
                    payload["reason"] = node_state.get("reason")
                elif node_name == "rewrite_query":
                    payload["web_query"] = node_state.get("web_query")
                elif node_name == "refine":
                    payload["total_strips_count"] = len(node_state.get("strips", []))
                    payload["kept_strips_count"] = len(node_state.get("kept_strips", []))
                elif node_name == "generate":
                    payload["answer"] = node_state.get("answer")
                    final_answer = node_state.get("answer")
                
                # Extract sources dynamically
                if "web_docs" in node_state:
                    web_src = [{"title": d.metadata.get("title"), "url": d.metadata.get("url")} for d in node_state["web_docs"]]
                    payload["web_sources"] = web_src
                    final_sources.extend([{"type": "Web", "title": s["title"], "url": s["url"]} for s in web_src])
                    
                if "good_docs" in node_state:
                    local_sources = []
                    for d in node_state["good_docs"]:
                        source_path = d.metadata.get("source", "Local Document")
                        filename = source_path.split("\\")[-1].split("/")[-1]
                        local_sources.append({"title": filename, "url": "#"})
                    payload["local_sources"] = local_sources
                    final_sources.extend([{"type": "Internal", "title": s["title"], "url": "#"} for s in local_sources])
                
                print(f"SSE YIELD: {payload}")
                
                yield {
                    "event": "update",
                    "data": json.dumps(payload)
                }
                
                await asyncio.sleep(0.01)

            # Once done, save the assistant message
            ast_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=final_answer,
                sources=json.dumps(final_sources)
            )
            db.add(ast_msg)
            db.commit()

            yield {
                "event": "done",
                "data": json.dumps({"status": "completed"})
            }
        except Exception as e:
            yield {
                "event": "error",
                "data": str(e)
            }

    return EventSourceResponse(event_generator())
