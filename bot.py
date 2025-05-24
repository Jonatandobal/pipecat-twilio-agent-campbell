import os
import sys
import json
import asyncio
from loguru import logger
from fastapi import WebSocket

from pipecat.frames.frames import LLMMessagesFrame, EndFrame, TextFrame
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
logger.add(sys.stderr, level="DEBUG")

async def run_bot(websocket_client: WebSocket, stream_sid: str, call_sid: str, testing: bool):
    """
    Función principal que ejecuta el bot de voz para Twilio
    """
    try:
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
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            audio_passthrough=True
        )

        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVEN_API_KEY"),
            voice_id=os.getenv("ELEVEN_VOICE_ID", "pNInz6obpgDQGcFmaJgB"),  # Adam voice por defecto
        )

        # Crear el contexto inicial de la conversación
        messages = [
            {
                "role": "system",
                "content": """You are Tasha, a helpful AI assistant speaking on a phone call.
                
                Keep your responses:
                - Very brief (1-2 sentences max)
                - Conversational and natural
                - Clear and easy to understand over the phone
                
                Start by greeting the caller and asking how you can help them.
                Wait for them to speak before responding.""",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        # Crear el pipeline
        pipeline = Pipeline([
            transport.input(),  # Audio de entrada de Twilio
            stt,                # Speech-to-text (Deepgram)
            context_aggregator.user(),  # Agregar mensaje del usuario al contexto
            llm,                # LLM (OpenAI)
            tts,                # Text-to-speech (ElevenLabs)
            transport.output(), # Audio de salida a Twilio
            context_aggregator.assistant(),  # Agregar respuesta del asistente al contexto
        ])

        # Variable para controlar si ya saludamos
        has_greeted = False

        # Configurar manejadores de eventos
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            nonlocal has_greeted
            logger.info("Client connected to Twilio bot")
            
            # Esperar un poco para que la conexión se estabilice
            await asyncio.sleep(3)
            
            if not has_greeted:
                # Enviar saludo inicial más directo
                greeting = "Hello! I'm Tasha, your AI assistant. How can I help you today?"
                await task.queue_frames([TextFrame(greeting)])
                has_greeted = True
                logger.info(f"Sent greeting: {greeting}")

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected from Twilio bot")
            await task.queue_frames([EndFrame()])

        # Crear y ejecutar la tarea del pipeline
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        logger.info("Starting Twilio bot pipeline")
        await PipelineRunner().run(task)

    except Exception as e:
        logger.error(f"Error in run_bot: {str(e)}", exc_info=True)
        raise
