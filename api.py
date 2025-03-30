import time
import asyncio
import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

from jose import JWTError
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect, Header, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from pydantic import ValidationError, BaseModel
from sqlalchemy.exc import SQLAlchemyError

from auth import verify_token, get_current_user
from config import SECRET_KEY, JWT_ALGORITHM, GEMINI_API_KEY, GEMINI_MODEL
from database import get_db, Message, ChatMessage, ConversationSummary, Conversation, User, Device
from models import MessageRequest, MessageResponse, ConversationRequest
from schemas import ChatRequest, ChatResponse, SyncRequest, SyncResponse
from exceptions import GeminiAPIError, AuthenticationError
import google.generativeai as genai
from rate_limiter import limiter, rate_limit
from cache_manager import cache_conversation, get_cached_conversation
from firestore_manager import store_document, get_document, update_document
from llm_service import generate_response, generate_streaming_response
from websocket_manager import manager

logger = logging.getLogger(__name__)

app = FastAPI(title="Eva LLM API")

router = APIRouter()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Start the heartbeat task
    asyncio.create_task(manager.broadcast_heartbeat())
    # Restore WebSocket connections from Firestore if implemented
    if hasattr(manager, 'restore_from_firestore'):
        asyncio.create_task(manager.restore_from_firestore())

# Initialize the Gemini client.
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)
chat_session = gemini_model.start_chat(history=[])

@router.post("/chat", response_model=MessageResponse)
@rate_limit(limit=20, period=60)  # 20 requests per minute
async def chat_endpoint(
    message_request: MessageRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    db: Session = Depends(get_db),
    auth_data: Dict[str, Any] = Depends(verify_token)
):
    device_id = auth_data.get("sub")  # "sub" contains user/device ID in JWT tokens
    
    # Register device if not exists
    device = db.query(Device).filter(Device.device_id == device_id).first()
    if not device:
        device = Device(device_id=device_id, device_name=f"Device-{device_id[:8]}")
        db.add(device)
        db.commit()
    else:
        # Update last sync time
        device.last_sync = datetime.utcnow()
        db.commit()
    
    # Get or create conversation
    conversation_id = message_request.conversation_id
    if conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conversation = Conversation(
            title=message_request.content[:50] if message_request.content else "New conversation",
            last_synced_device=device_id
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        conversation_id = conversation.id
    
    # Update conversation metadata
    conversation.last_synced_device = device_id
    conversation.updated_at = datetime.utcnow()
    db.commit()
    
    # Get conversation history
    conversation_history = await get_cached_conversation(str(conversation_id))
    if not conversation_history:
        messages = db.query(ChatMessage).filter(
            ChatMessage.conversation_id == conversation_id
        ).order_by(ChatMessage.created_at).all()
        
        conversation_history = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
    
    # Save user message
    user_message = ChatMessage(
        conversation_id=conversation_id,
        role="user",
        content=message_request.content,
        device_id=device_id
    )
    db.add(user_message)
    db.commit()
    
    # Add user message to history
    conversation_history.append({"role": "user", "content": message_request.content})
    
    # Generate response using the LLM
    llm_response = generate_response(conversation_history, message_request.model)
    
    # Save assistant message
    assistant_message = ChatMessage(
        conversation_id=conversation_id,
        role="assistant",
        content=llm_response,
        device_id="system"
    )
    db.add(assistant_message)
    db.commit()
    
    # Update conversation history and cache it
    conversation_history.append({"role": "assistant", "content": llm_response})
    background_tasks.add_task(cache_conversation, str(conversation_id), conversation_history)
    
    # Broadcast to other connected devices
    background_tasks.add_task(
        manager.broadcast_to_conversation,
        str(conversation_id),
        {
            "type": "new_message",
            "conversation_id": str(conversation_id),
            "message": {
                "role": "assistant",
                "content": llm_response,
                "created_at": datetime.utcnow().isoformat()
            }
        }
    )
    
    return MessageResponse(
        conversation_id=str(conversation_id),
        message=llm_response,
        created_at=datetime.utcnow().isoformat()
    )

@app.get("/api/conversations", response_model=List[ConversationRequest])
async def get_conversations(
    db: Session = Depends(get_db),
    auth_data: Dict[str, Any] = Depends(verify_token)
):
    device_id = auth_data.get("sub")
    
    # Register device if not exists
    device = db.query(Device).filter(Device.device_id == device_id).first()
    if not device:
        device = Device(device_id=device_id, device_name=f"Device-{device_id[:8]}")
        db.add(device)
        db.commit()
    
    # Get all conversations
    conversations = db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
    
    return [
        ConversationRequest(
            id=str(conv.id),
            title=conv.title,
            created_at=conv.created_at.isoformat(),
            updated_at=conv.updated_at.isoformat()
        ) for conv in conversations
    ]

@app.post("/api/voice", response_model=MessageResponse)
@rate_limit(limit=5, period=60)  # 5 requests per minute - voice is more resource intensive
async def voice_endpoint(
    message_request: MessageRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    db: Session = Depends(get_db),
    user_data: dict = Depends(verify_token)
):
    # This is the same as chat_endpoint but could be optimized for voice interactions
    # For now, we'll just call the chat endpoint
    return await chat_endpoint(message_request, background_tasks, request, db, user_data)

@app.websocket("/ws/voice/{conversation_id}")
async def voice_websocket(
    websocket: WebSocket, 
    conversation_id: str,
    device_id: str = None,
    token: str = None
):
    if not device_id or not token:
        await websocket.close(code=1008, reason="Missing device authentication")
        return
    
    # Replace the direct token check with JWT verification
    try:
        # Verify token and extract device_id from it
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        token_device_id = payload.get("sub")
        
        # Verify the provided device_id matches the token
        if token_device_id != device_id:
            await websocket.close(code=1008, reason="Device ID mismatch")
            return
    except JWTError:
        await websocket.close(code=1008, reason="Invalid token")
        return
    
    # Accept the connection
    await manager.connect(websocket, device_id)
    await manager.join_conversation(conversation_id, device_id)
    
    # Get DB session
    db = next(get_db())
    
    try:
        # Notify other devices about this connection
        await manager.broadcast_to_conversation(
            conversation_id,
            {
                "type": "device_connected",
                "device_id": device_id,
                "conversation_id": conversation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Process messages
        while True:
            # Receive message from WebSocket
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle different message types
            message_type = message_data.get("type", "")
            
            if message_type == "voice_input":
                # Process voice input (text transcribed from voice)
                user_input = message_data.get("content", "")
                
                # Get conversation history
                conversation_history = await get_cached_conversation(conversation_id)
                if not conversation_history:
                    messages = db.query(ChatMessage).filter(
                        ChatMessage.conversation_id == conversation_id
                    ).order_by(ChatMessage.created_at).all()
                    
                    conversation_history = [
                        {"role": msg.role, "content": msg.content} for msg in messages
                    ]
                
                # Save user message
                user_message = ChatMessage(
                    conversation_id=uuid.UUID(conversation_id),
                    role="user",
                    content=user_input,
                    device_id=device_id
                )
                db.add(user_message)
                db.commit()
                
                # Add to history
                conversation_history.append({"role": "user", "content": user_input})
                
                # Broadcast user message to other devices
                await manager.broadcast_to_conversation(
                    conversation_id,
                    {
                        "type": "new_message",
                        "conversation_id": conversation_id,
                        "message": {
                            "role": "user",
                            "content": user_input,
                            "device_id": device_id,
                            "created_at": datetime.utcnow().isoformat()
                        }
                    }
                )
                
                # Initialize full response for saving to database
                full_response = ""
                
                # Generate streaming response
                async def stream_callback(chunk):
                    await manager.broadcast_to_conversation(
                        conversation_id,
                        {
                            "type": "voice_chunk",
                            "conversation_id": conversation_id,
                            "content": chunk,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                
                # Stream the response
                async for chunk in generate_streaming_response(
                    conversation_history, 
                    message_data.get("model", "gemini-1.5-flash"),
                    stream_callback
                ):
                    full_response += chunk
                
                # Save the complete assistant response to database
                assistant_message = ChatMessage(
                    conversation_id=uuid.UUID(conversation_id),
                    role="assistant",
                    content=full_response,
                    device_id="system"
                )
                db.add(assistant_message)
                db.commit()
                
                # Update conversation history and cache
                conversation_history.append({"role": "assistant", "content": full_response})
                await cache_conversation(conversation_id, conversation_history)
                
                # Update conversation metadata
                conversation = db.query(Conversation).filter(
                    Conversation.id == uuid.UUID(conversation_id)
                ).first()
                if conversation:
                    conversation.last_synced_device = device_id
                    conversation.updated_at = datetime.utcnow()
                    db.commit()
                
                # Send complete message notification
                await manager.broadcast_to_conversation(
                    conversation_id,
                    {
                        "type": "voice_response_complete",
                        "conversation_id": conversation_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
            elif message_type == "heartbeat":
                # Respond to heartbeat
                await websocket.send_json({
                    "type": "heartbeat_ack",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            elif message_type == "leave":
                # Leave conversation
                break
                
    except WebSocketDisconnect:
        # Handle disconnection
        await manager.disconnect(device_id)
        await manager.leave_conversation(conversation_id, device_id)
        await manager.broadcast_to_conversation(
            conversation_id,
            {
                "type": "device_disconnected",
                "device_id": device_id,
                "conversation_id": conversation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    finally:
        # Clean up
        db.close()
        await manager.disconnect(device_id)
        await manager.leave_conversation(conversation_id, device_id)

# Create a helper function to verify devices using Firestore
async def verify_device(device_id: str) -> bool:
    """Verify a device is registered and authorized using Firestore"""
    try:
        device_data = await get_document("devices", device_id)
        if not device_data:
            logger.warning(f"Unauthorized device: {device_id}")
            return False
        
        if not device_data.get("verified"):
            logger.warning(f"Device not verified: {device_id}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error verifying device {device_id}: {e}")
        return False

@router.post("/chat", response_model=ChatResponse)
@limiter.limit("5/minute")
async def chat_endpoint(
    request: Request,
    request_data: ChatRequest,
    token: str = Depends(verify_token),
    db: AsyncSession = Depends(get_db)
) -> ChatResponse:
    """
    Handle chat requests by sending user text to the Gemini API.
    Uses token-based authentication.
    """
    try:
        logger.debug(f"Received chat request with text: {request_data.text}")
        user_text = request_data.text.strip()
        if not user_text:
            raise HTTPException(status_code=400, detail="Empty message not allowed.")
        
        # Check that the device is verified using Firestore
        device_data = await get_document("devices", request_data.device_id)
        if not device_data:
            logger.warning(f"Unauthorized device: {request_data.device_id}")
            raise AuthenticationError(detail="Device not authorized")
        
        if not device_data.get("verified"):
            logger.warning(f"Device not verified: {request_data.device_id}")
            raise AuthenticationError(detail="Device not authorized")
        
        timestamp = int(time.time() * 1000)
        
        # Save the user message.
        new_message = ChatMessage(message=user_text, is_user=True, timestamp=timestamp)
        db.add(new_message)
        
        # Retrieve conversation summaries.
        scalar_result = await db.execute(select(ConversationSummary))
        summaries = scalar_result.scalars().all()
        aggregated_memory = "; ".join([f"{s.label}: {s.summary}" for s in summaries]) if summaries else ""
        logger.debug(f"Aggregated memory: {aggregated_memory}")
        
        # Build the prompt for the Gemini API.
        prompt = f"Remember: {aggregated_memory}\nUser: {user_text}" if aggregated_memory else f"User: {user_text}"
        logger.debug(f"Constructed prompt: {prompt}")
        
        # Call the Gemini API asynchronously.
        ai_response = await call_gemini_text_async(prompt)
        logger.debug(f"Received AI response: {ai_response}")
        
        # Save the assistant's message.
        new_assistant_message = ChatMessage(message=ai_response, is_user=False, timestamp=timestamp)
        db.add(new_assistant_message)
        await db.commit()
        logger.info("Chat messages saved successfully.")
        
        return ChatResponse(answer=ai_response, memory_updated=False)
    except HTTPException as http_ex:
        raise http_ex
    except ValidationError as validation_error:
        logger.error(f"Validation Error: {validation_error}", exc_info=True)
        raise HTTPException(status_code=422, detail=validation_error.errors()) from validation_error
    except SQLAlchemyError as e:
        logger.error(f"Database error in chat endpoint: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}") from e
    except GeminiAPIError as e:
        logger.error(f"Gemini API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}") from e
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error in chat endpoint") from e

async def call_gemini_text_async(prompt: str) -> str:
    """Call the Gemini API for a text response asynchronously."""
    logger.debug(f"Calling Gemini API with prompt: {prompt}")
    try:
        response = await asyncio.to_thread(chat_session.send_message, prompt)
        answer = response.text
        logger.debug(f"Gemini API returned answer: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}", exc_info=True)
        raise GeminiAPIError(detail=f"Error calling Gemini API: {e}")

async def periodic_memory_sync() -> None:
    """
    Periodically synchronize conversation memory.
    """
    while True:
        logger.debug("Running periodic memory sync.")
        await asyncio.sleep(3600)

@router.post("/sync", response_model=SyncResponse)
async def sync_device(
    sync_request: SyncRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Synchronize data between device and backend
    """
    try:
        # Get last sync timestamp from device
        last_sync = sync_request.last_sync_timestamp
        
        # Fetch conversations updated since last sync
        updated_conversations = await db.get_conversations_since(
            user_id=current_user.id,
            since_timestamp=last_sync
        )
        
        # Fetch user settings updated since last sync
        updated_settings = await db.get_settings_since(
            user_id=current_user.id,
            since_timestamp=last_sync
        )
        
        # Process any pending updates from the device
        if sync_request.local_changes:
            await db.apply_device_changes(
                user_id=current_user.id,
                device_id=sync_request.device_id,
                changes=sync_request.local_changes
            )
        
        # Get current server timestamp for next sync
        current_timestamp = datetime.utcnow().isoformat()
        
        return SyncResponse(
            conversations=updated_conversations,
            settings=updated_settings,
            server_timestamp=current_timestamp
        )
    except Exception as e:
        logger.error(f"Sync error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Synchronization failed"
        )