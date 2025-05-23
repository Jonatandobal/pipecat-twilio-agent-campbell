#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import datetime
import io
import os
import sys
import wave

import aiofiles
from dotenv import load_dotenv
from fastapi import WebSocket
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)

# Configuración de logging
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")
logger.add("pipecat_debug.log", rotation="10 MB", level="DEBUG")


async def save_audio(server_name: str, audio: bytes, sample_rate: int, num_channels: int):
    if len(audio) > 0:
        filename = (
            f"{server_name}_recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Merged audio saved to {filename}")
    else:
        logger.info("No audio data to save")


async def run_bot(websocket_client: WebSocket, stream_sid: str, testing: bool):
    try:
        logger.info(f"Iniciando sesión con stream_sid: {stream_sid}")
        
        # Configuración del transporte siguiendo el ejemplo funcional
        transport = FastAPIWebsocketTransport(
            websocket=websocket_client,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),  # Sin parámetros adicionales
                vad_audio_passthrough=True,
                serializer=TwilioFrameSerializer(stream_sid),
            ),
        )

        # Configuración del modelo LLM
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"), 
            model="gpt-4o-mini",
            temperature=0.7,
        )

        # Configuración de STT para español
        stt = OpenAISTTService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="whisper-1",
            language="es",
        )

        # Configuración de TTS
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVEN_API_KEY"),
            voice_id=os.getenv("ELEVEN_VOICE_ID"),
            model_id="eleven_multilingual_v2",  # Modelo multilingüe para mejor español
            optimize_streaming_latency=4,
        )

        # Mensajes del sistema en español
        messages = [
            {
                "role": "system",
                "content": """
                Eres Juan, asesor de facturación eléctrica. Hablas español profesional y empático.
                Tu trabajo es verificar datos del cliente Pedro Martinez Garcia de Madrid para actualizar su convenio eléctrico.
                No vendes nada, solo actualizas información. Mantén respuestas cortas y claras.
                
                IMPORTANTE:
                - SIEMPRE responde en español.
                - Usa frases cortas y naturales.
                - Espera que el cliente responda antes de continuar.
                - Si no entiendes algo, pide amablemente que lo repita.
                - Mantén un tono conversacional y amigable.
                """,
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        audiobuffer = AudioBufferProcessor(user_continuous_stream=not testing)

        # Configuración del pipeline siguiendo el ejemplo funcional
        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                audiobuffer,
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=8000,
                audio_out_sample_rate=8000,
                allow_interruptions=True,
            ),
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            try:
                logger.info("Cliente conectado. Iniciando grabación y presentación.")
                await audiobuffer.start_recording()
                
                # Mensaje inicial en español
                messages.append({
                    "role": "system", 
                    "content": "Preséntate al usuario en español. Di: 'Hola, soy Juan, asesor de facturación eléctrica. ¿En qué puedo ayudarte hoy?'"
                })
                
                await task.queue_frames([context_aggregator.user().get_context_frame()])
                logger.info("Mensaje inicial enviado correctamente")
            except Exception as e:
                logger.error(f"Error en on_client_connected: {str(e)}")

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Cliente desconectado. Cancelando tarea.")
            await task.cancel()

        @audiobuffer.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio, sample_rate, num_channels):
            server_name = f"server_{websocket_client.client.port}"
            await save_audio(server_name, audio, sample_rate, num_channels)

        # Manejadores de eventos para depuración
        @stt.event_handler("on_transcription")
        async def on_transcription(service, text, metadata=None):
            logger.info(f"Transcripción STT: {text}")
            if metadata:
                logger.debug(f"Metadata STT: {metadata}")

        @llm.event_handler("on_response")
        async def on_llm_response(service, response):
            logger.info(f"Respuesta LLM: {response}")

        @tts.event_handler("on_audio")
        async def on_tts_audio(service, audio_data):
            logger.info(f"Audio TTS generado: {len(audio_data)} bytes")

        runner = PipelineRunner(handle_sigint=False, force_gc=True)
        logger.info("Iniciando ejecución del pipeline")
        await runner.run(task)
        
    except Exception as e:
        logger.error(f"Error en run_bot: {str(e)}")
        # Intentamos cerrar la conexión de forma limpia
        try:
            if 'task' in locals():
                await task.cancel()
        except:
            pass
