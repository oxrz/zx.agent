"""
zAgent text-to-speech module (optional)
"""

import asyncio
import subprocess
import tempfile
from pathlib import Path
from utils.logger import logger


class TextToSpeech:
    def __init__(self, provider="edge-tts", voice="zh-CN-XiaoxiaoNeural", speed=1.0):
        self.provider = provider
        self.voice = voice
        self.speed = speed

    def speak(self, text):
        if self.provider == "edge-tts":
            return self._speak_edge_tts(text)
        elif self.provider == "gtts":
            return self._speak_gtts(text)
        else:
            logger.error(f"Unsupported TTS provider: {self.provider}")
            return False

    def _speak_edge_tts(self, text):
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, self.voice, rate=f"+{int((self.speed - 1) * 100)}%")
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_path = temp_file.name
            temp_file.close()
            asyncio.run(communicate.save(temp_path))
            subprocess.run(["ffplay", "-nodisp", "-autoexit", temp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            Path(temp_path).unlink(missing_ok=True)
            return True
        except ImportError:
            logger.warning("edge-tts is not installed, run: pip install edge-tts")
            return False
        except Exception as e:
            logger.error(f"edge-tts playback failed: {e}")
            return False

    def _speak_gtts(self, text):
        try:
            from gtts import gTTS
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_path = temp_file.name
            temp_file.close()
            tts = gTTS(text=text, lang="zh-CN", slow=False)
            tts.save(temp_path)
            subprocess.run(["ffplay", "-nodisp", "-autoexit", temp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            Path(temp_path).unlink(missing_ok=True)
            return True
        except ImportError:
            logger.warning("gtts is not installed, run: pip install gtts")
            return False
        except Exception as e:
            logger.error(f"gTTS playback failed: {e}")
            return False
