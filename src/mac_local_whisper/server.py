"""Voice server: listens on a Unix socket for toggle commands to record and transcribe."""

import atexit
import json
import os
import signal
import socket
import sys

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SOCKET_PATH = "/tmp/mac-local-whisper.sock"
SAMPLE_RATE = 16000
CHANNELS = 1
MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class VoiceServer:
    def __init__(self) -> None:
        self.recording = False
        self.audio_chunks: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self.model: WhisperModel | None = None
        self.sock: socket.socket | None = None

    def load_model(self) -> None:
        log(f"Loading whisper model '{MODEL_SIZE}' (compute_type={COMPUTE_TYPE})...")
        self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
        log("Model loaded.")

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags
    ) -> None:
        if status:
            log(f"Audio callback status: {status}")
        self.audio_chunks.append(indata.copy())

    def start_recording(self) -> dict:
        self.audio_chunks = []
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.stream.start()
        self.recording = True
        log("Recording...")
        return {"status": "recording"}

    def stop_and_transcribe(self) -> dict:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.recording = False
        log("Recording stopped. Transcribing...")

        if not self.audio_chunks:
            log("No audio captured.")
            return {"status": "done", "text": ""}

        audio = np.concatenate(self.audio_chunks, axis=0).flatten()
        segments, info = self.model.transcribe(audio, beam_size=5, language="en")
        text = " ".join(seg.text.strip() for seg in segments).strip()

        log(f"Transcribed: {text}")
        return {"status": "done", "text": text}

    def handle_toggle(self) -> dict:
        if not self.recording:
            return self.start_recording()
        else:
            return self.stop_and_transcribe()

    def cleanup(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        if self.sock is not None:
            self.sock.close()
        try:
            os.unlink(SOCKET_PATH)
        except FileNotFoundError:
            pass

    def serve(self) -> None:
        try:
            os.unlink(SOCKET_PATH)
        except FileNotFoundError:
            pass

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(SOCKET_PATH)
        self.sock.listen(1)
        os.chmod(SOCKET_PATH, 0o666)

        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

        log(f"Listening on {SOCKET_PATH}")

        while True:
            conn, _ = self.sock.accept()
            try:
                data = conn.recv(1024).decode("utf-8").strip()
                if data == "toggle":
                    response = self.handle_toggle()
                else:
                    response = {"error": f"unknown command: {data}"}
                conn.sendall(json.dumps(response).encode("utf-8"))
            except Exception as e:
                log(f"Error handling connection: {e}")
                try:
                    conn.sendall(json.dumps({"error": str(e)}).encode("utf-8"))
                except Exception:
                    pass
            finally:
                conn.close()


def main() -> None:
    server = VoiceServer()
    server.load_model()
    server.serve()
