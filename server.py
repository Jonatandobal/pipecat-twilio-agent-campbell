import argparse
import json
import os
import uvicorn
from bot import run_bot
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
import logging

# Configurar logging exacto como el tutorial
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Pipecat Twilio Agent is running"}

@app.post("/")
async def start_call(request: Request):
    """Exacto como el tutorial - servir streams.xml"""
    logger.info("POST TwiML request received")
    try:
        return HTMLResponse(
            content=open("templates/streams.xml").read(), 
            media_type="application/xml"
        )
    except Exception as e:
        logger.error(f"Error serving TwiML: {str(e)}")
        # Fallback básico
        fallback_twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://pipecat-twilio-agent-campbell-production.up.railway.app/ws"></Stream>
  </Connect>
  <Pause length="60"/>
</Response>"""
        return HTMLResponse(content=fallback_twiml, media_type="application/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Exacto como el tutorial"""
    logger.info("WebSocket connection attempt")
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    # Exacto como el tutorial
    start_data = websocket.iter_text()
    first_message = await start_data.__anext__()
    logger.info(f"Received first message: {first_message}")
    
    # Segundo mensaje con los datos
    call_data = json.loads(await start_data.__anext__())
    logger.info(f"Call data: {call_data}")
    
    stream_sid = call_data["start"]["streamSid"]
    call_sid = call_data["start"]["callSid"]
    logger.info(f"Stream SID: {stream_sid}")
    logger.info(f"Call SID: {call_sid}")
    
    # Ejecutar bot exacto como tutorial
    await run_bot(websocket, stream_sid, call_sid, app.state.testing)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Twilio Chatbot Server")
    parser.add_argument(
        "-t", "--test", action="store_true", default=False, 
        help="set the server in testing mode"
    )
    args, *_ = parser.parse_known_args()
    app.state.testing = args.test
    
    # Railway port
    port = int(os.environ.get("PORT", 8765))  # Cambiado a 8765 como el tutorial
    
    logger.info(f"Starting server on 0.0.0.0:{port}, testing mode: {app.state.testing}")
    uvicorn.run(app, host="0.0.0.0", port=port)
