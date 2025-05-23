import os
import sys
import json
from loguru import logger
from fastapi import WebSocket

from pipecat.frames.frames import LLMMessagesFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import OpenAILLMService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.vad.silero import SileroVADAnalyzer

# ESTO ES LO IMPORTANTE: Usar FastAPI WebSocket Transport en lugar de Daily
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

async def run_bot(websocket_client: WebSocket, stream_sid: str, testing: bool):
    """
    Función principal que ejecuta el bot de voz para Twilio
    """
    try:
        # Inicializar el serializador de Twilio con el stream SID
        serializer = TwilioFrameSerializer(
            stream_sid=stream_sid,
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
                "content": """You are Tasha, a helpful AI assistant. You are speaking on a phone call.
                
                Keep your responses:
                - Conversational and natural
                - Brief and to the point
                - Friendly and professional
                
                You can help with general questions, provide information, and have casual conversations.
                Remember you're on a phone call, so speak clearly and don't use special formatting.""",
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

        # Configurar manejadores de eventos
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected to Twilio bot")
            # Enviar mensaje de bienvenida
            messages = [
                {
                    "role": "system",
                    "content": "The user just connected. Greet them as Tasha and ask how you can help them today."
                }
            ]
            await task.queue_frames([LLMMessagesFrame(messages)])

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
