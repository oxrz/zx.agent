"""
zAgent conversation/context memory module
"""

import time
from typing import List, Dict


class ContextBuffer:
    """
    Rolling statement context cache (used by assist mode).

    Different from a typical ConversationMemory: ConversationMemory stores "what the
    user asked / what the AI answered" turn by turn. ContextBuffer instead stores the
    raw text of statements heard over a recent time window (not turn count), used as
    background when answering later questions (e.g. something said 20 minutes ago in
    a meeting is probably still relevant, but something said 2 hours ago probably
    isn't related to the current topic anymore).
    """

    def __init__(self, window_seconds: float = 1200.0):
        # 20-minute (1200s) rolling window by default
        self.window_seconds = window_seconds
        self._items: List[Dict] = []  # [{"ts": float, "text": str, "source": str}]

    def add(self, text: str, source: str = ""):
        if not text:
            return
        self._items.append({"ts": time.time(), "text": text, "source": source})
        self._trim()

    def _trim(self):
        cutoff = time.time() - self.window_seconds
        self._items = [item for item in self._items if item["ts"] >= cutoff]

    def get_context_text(self) -> str:
        """Return the text within the window, concatenated in chronological order, for use in the LLM prompt"""
        self._trim()
        return " ".join(item["text"] for item in self._items)

    def clear(self):
        self._items = []

    def __len__(self):
        self._trim()
        return len(self._items)

    def __repr__(self):
        return f"ContextBuffer(items={len(self)}, window={self.window_seconds}s)"
