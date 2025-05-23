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
    return {"status": "ok", "message": "Pipecat Twilio Agent is running"}

@app.post("/")
async def start_call(request: Request):
    logger.info("POST TwiML request received")
    try:
        # Check if Railway URL is available
        railway_url = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
        
        if railway_url and os.path.exists("templates/streams.xml"):
            # Read the template
            with open("templates/streams.xml", "r") as file:
                template_content = file.read()
            
            # If we're running in Railway, ensure the WebSocket URL is correct
            if railway_url and "<Stream url=" in template_content:
                # Replace the WebSocket URL with the Railway URL
                ws_url = f"wss://{railway_url}/ws"
                template_content = template_content.replace(
                    "<Stream url=\"ws://localhost:8765/ws\"></Stream>", 
                    f"<Stream url=\"{ws_url}\"></Stream>"
                )
                logger.info(f"Updated WebSocket URL to: {ws_url}")
                
            return HTMLResponse(content=template_content, media_type="application/xml")
        else:
            # Fallback to just reading the file
            return HTMLResponse(content=open("templates/streams.xml").read(), media_type="application/xml")
    except Exception as e:
        logger.error(f"Error serving TwiML: {str(e)}")
        # Create a basic TwiML response as fallback
        fallback_twiml = """
        <Response>
          <Connect>
            <Stream url="wss://RAILWAY_URL/ws"></Stream>
          </Connect>
          <Pause length="40"/>
        </Response>
        """
        return HTMLResponse(content=fallback_twiml, media_type="application/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
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
        logger.error(f"Error in WebSocket endpoint: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Twilio Chatbot Server")
    parser.add_argument(
        "-t", "--test", action="store_true", default=False, help="set the server in testing mode"
    )
    args, *_ = parser.parse_known_args()
    app.state.testing = args.test
    
    # Use Railway's PORT environment variable or default to 8765
    port = int(os.environ.get("PORT", 8765))
    
    logger.info(f"Starting server on port {port}, testing mode: {app.state.testing}")
    uvicorn.run(app, host="0.0.0.0", port=port)
