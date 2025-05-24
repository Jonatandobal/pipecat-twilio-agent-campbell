import os
import sys
import asyncio
from loguru import logger
from fastapi import WebSocket

from pipecat.frames.frames import LLMMessagesFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

# Importaciones exactas como el tutorial
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.audio.vad.silero import SileroVADAnalyzer

from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

async def run_bot(websocket: WebSocket, stream_sid: str, call_sid: str, testing: bool):
    """
    Bot exactamente como el tutorial original pero adaptado para Railway
    """
    try:
        logger.info(f"Starting bot - stream_sid: {stream_sid}, call_sid: {call_sid}")

        # Serializador como en el tutorial original
        serializer = TwilioFrameSerializer(
            stream_sid=stream_sid,
            call_sid=call_sid,
            account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
            auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
        )

        # Transport exacto como el tutorial
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                vad_analyzer=SileroVADAnalyzer(),
                serializer=serializer,
            ),
        )

        # Servicios exactos como el tutorial
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini"
        )

        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            audio_passthrough=True
        )

        # ElevenLabs como en el tutorial original
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVEN_API_KEY"),
            voice_id=os.getenv("ELEVEN_VOICE_ID", "pNInz6obpgDQGcFmaJgB"),
        )

        # Mensaje del sistema exacto como el tutorial (línea 83 del video)
        messages = [
            {
                "role": "system",
                "content": "Hey you're a helpful assistant named Tasha and some other stuff",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        # Pipeline exacto como el tutorial
        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])

        # Task como el tutorial
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        # Event handlers como el tutorial
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            # El tutorial parece enviar un saludo automático
            await task.queue_frames([LLMMessagesFrame([
                {
                    "role": "system", 
                    "content": "Please introduce yourself to the user."
                }
            ])])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await task.queue_frames([EndFrame()])

        # Ejecutar exacto como el tutorial
        logger.info("Starting pipeline")
        await PipelineRunner().run(task)

    except Exception as e:
        logger.error(f"Error in run_bot: {str(e)}", exc_info=True)
        raise
