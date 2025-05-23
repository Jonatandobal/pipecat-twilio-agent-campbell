#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
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

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    """Healthcheck endpoint para Railway"""
    logger.info("Health check endpoint called")
    return {"status": "ok", "message": "Pipecat Twilio Agent is running", "service": "healthy"}

@app.get("/health")
async def detailed_health():
    """Endpoint de salud detallado"""
    try:
        # Verificar que los archivos necesarios existen
        files_status = {
            "streams_xml": os.path.exists("templates/streams.xml"),
            "streams_template": os.path.exists("templates/streams.xml.template")
        }
        
        # Verificar variables de entorno críticas
        env_status = {
            "PORT": os.environ.get("PORT", "Not set"),
            "RAILWAY_STATIC_URL": os.environ.get("RAILWAY_STATIC_URL", "Not set"),
            "OPENAI_API_KEY": "Set" if os.environ.get("OPENAI_API_KEY") else "Not set"
        }
        
        return {
            "status": "healthy",
            "service": "pipecat-twilio-agent",
            "files": files_status,
            "environment": env_status
        }
    except Exception as e:
        logger.error(f"Error in detailed health check: {str(e)}")
        return {"status": "error", "message": str(e)}

def create_fallback_twiml(ws_url: str = None) -> str:
    """Crear TwiML de respaldo"""
    if not ws_url:
        # Intentar obtener la URL de Railway
        railway_url = os.environ.get("RAILWAY_STATIC_URL")
        if railway_url:
            ws_url = f"wss://{railway_url}/ws"
        else:
            ws_url = "wss://localhost:8765/ws"
    
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}"></Stream>
  </Connect>
  <Pause length="40"/>
</Response>"""

@app.post("/")
async def start_call(request: Request):
    """Endpoint para manejar llamadas de Twilio"""
    logger.info("POST TwiML request received")
    try:
        # Obtener la URL de Railway
        railway_url = os.environ.get("RAILWAY_STATIC_URL")
        logger.info(f"Railway URL: {railway_url}")
        
        # Intentar leer el archivo streams.xml
        streams_file_path = "templates/streams.xml"
        template_file_path = "templates/streams.xml.template"
        
        template_content = None
        
        # Intentar leer streams.xml primero
        if os.path.exists(streams_file_path):
            logger.info("Reading streams.xml file")
            with open(streams_file_path, "r") as file:
                template_content = file.read()
        elif os.path.exists(template_file_path):
            logger.info("Reading streams.xml.template file")
            with open(template_file_path, "r") as file:
                template_content = file.read()
        else:
            logger.warning("No streams.xml or template file found, using fallback")
        
        # Si tenemos contenido del template y URL de Railway, actualizamos
        if template_content and railway_url:
            ws_url = f"wss://{railway_url}/ws"
            logger.info(f"Updating WebSocket URL to: {ws_url}")
            
            # Reemplazar varias posibles URLs en el template
            replacements = [
                ("ws://localhost:8765/ws", ws_url),
                ("wss://localhost:8765/ws", ws_url),
                ("<your server url>", railway_url),
                ("ws://<your server url>/ws", ws_url),
                ("wss://<your server url>/ws", ws_url)
            ]
            
            for old, new in replacements:
                template_content = template_content.replace(old, new)
        
        # Si no tenemos template, crear uno de respaldo
        if not template_content:
            logger.info("Creating fallback TwiML")
            template_content = create_fallback_twiml()
        
        logger.info(f"Returning TwiML: {template_content}")
        return HTMLResponse(content=template_content, media_type="application/xml")
        
    except Exception as e:
        logger.error(f"Error serving TwiML: {str(e)}", exc_info=True)
        # Crear respuesta TwiML de emergencia
        fallback_twiml = create_fallback_twiml()
        return HTMLResponse(content=fallback_twiml, media_type="application/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint WebSocket para manejar las llamadas"""
    logger.info("WebSocket connection attempt")
    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")
        
        start_data = websocket.iter_text()
        first_message = await start_data.__anext__()
        logger.info(f"Received first message: {first_message}")
        
        # Parse the second message which contains the stream SID
        try:
            call_data = json.loads(await start_data.__anext__())
            logger.info(f"Call data: {call_data}")
            stream_sid = call_data["start"]["streamSid"]
            logger.info(f"Stream SID: {stream_sid}")
        except Exception as e:
            logger.error(f"Error parsing call data: {str(e)}")
            stream_sid = "unknown_sid"
        
        # Run the bot
        await run_bot(websocket, stream_sid, app.state.testing)
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint: {str(e)}", exc_info=True)

@app.on_event("startup")
async def startup_event():
    """Evento al iniciar la aplicación"""
    logger.info("Application starting up")
    logger.info(f"PORT: {os.environ.get('PORT', 'Not set')}")
    logger.info(f"RAILWAY_STATIC_URL: {os.environ.get('RAILWAY_STATIC_URL', 'Not set')}")
    
    # Verificar archivos necesarios
    files_to_check = ["templates/streams.xml", "templates/streams.xml.template"]
    for file_path in files_to_check:
        if os.path.exists(file_path):
            logger.info(f"✓ Found: {file_path}")
        else:
            logger.warning(f"✗ Missing: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Twilio Chatbot Server")
    parser.add_argument(
        "-t", "--test", action="store_true", default=False, help="set the server in testing mode"
    )
    args, *_ = parser.parse_known_args()
    app.state.testing = args.test
    
    # Use Railway's PORT environment variable or default to 8765
    port = int(os.environ.get("PORT", 8765))
    
    logger.info(f"Starting server on 0.0.0.0:{port}, testing mode: {app.state.testing}")
    
    # Mostrar información del entorno al inicio
    logger.info(f"Environment variables:")
    logger.info(f"  - PORT: {os.environ.get('PORT', 'Not set')}")
    logger.info(f"  - RAILWAY_STATIC_URL: {os.environ.get('RAILWAY_STATIC_URL', 'Not set')}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
