import os
import sys
import json
import asyncio
from loguru import logger
from fastapi import WebSocket

from pipecat.frames.frames import LLMMessagesFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

# Importaciones actualizadas para Pipecat 0.0.67
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.audio.vad.silero import SileroVADAnalyzer

# ESTO ES LO IMPORTANTE: Usar FastAPI WebSocket Transport en lugar de Daily
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer

logger.remove(0)
logger.add(sys.stderr, level="INFO")  # Cambié a INFO para menos spam

async def run_bot(websocket_client: WebSocket, stream_sid: str, call_sid: str, testing: bool):
    """
    Función principal que ejecuta el bot de voz para Twilio
    """
    try:
        logger.info(f"Starting bot with stream_sid: {stream_sid}, call_sid: {call_sid}")
        
        # Inicializar el serializador de Twilio con todos los parámetros necesarios
        serializer = TwilioFrameSerializer(
            stream_sid=stream_sid,
            call_sid=call_sid,
            account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
            auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
        )

        # Crear el transporte WebSocket de FastAPI
        transport = FastAPIWebsocketTransport(
            websocket=websocket_client,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                vad_analyzer=SileroVADAnalyzer(),
                serializer=serializer,
            ),
        )

        # Inicializar servicios de IA
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini"
        )

        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY")
        )

        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVEN_API_KEY"),
            voice_id=os.getenv("ELEVEN_VOICE_ID", "pNInz6obpgDQGcFmaJgB"),
        )

        # Crear el contexto inicial de la conversación
        messages = [
            {
                "role": "system",
                "content": "You are Tasha, a helpful AI assistant. You're on a phone call. Keep responses very short and conversational.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        # Crear el pipeline
        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])

        # Crear y ejecutar la tarea del pipeline
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
        )

        # Variable para controlar el saludo
        greeted = False

        # Configurar manejadores de eventos
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            nonlocal greeted
            logger.info("Client connected - waiting before greeting")
            
            # Esperar más tiempo para que todo se estabilice
            await asyncio.sleep(4)
            
            if not greeted:
                logger.info("Sending greeting message")
                greeting_messages = [
                    {
                        "role": "user",
                        "content": "Say hello and introduce yourself as Tasha"
                    }
                ]
                await task.queue_frames([LLMMessagesFrame(greeting_messages)])
                greeted = True

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected from bot")
            await task.queue_frames([EndFrame()])

        logger.info("Starting pipeline")
        await PipelineRunner().run(task)

    except Exception as e:
        logger.error(f"Error in run_bot: {str(e)}", exc_info=True)
        raise
