"""元に戻す / やり直し。編集状態全体のスナップショットを積む単純な方式。"""

from __future__ import annotations

import copy
from typing import Any


class History:
    def __init__(self, limit: int = 200):
        self.limit = limit
        self._undo: list[Any] = []
        self._redo: list[Any] = []
        self.current: Any = None

    def reset(self, state: Any) -> None:
        self._undo.clear()
        self._redo.clear()
        self.current = copy.deepcopy(state)

    def commit(self, state: Any) -> bool:
        """状態が変わっていれば履歴に積む。積んだら True。"""
        if state == self.current:
            return False
        if self.current is not None:
            self._undo.append(self.current)
            del self._undo[: -self.limit]
        self.current = copy.deepcopy(state)
        self._redo.clear()
        return True

    def undo(self) -> Any | None:
        if not self._undo:
            return None
        self._redo.append(self.current)
        self.current = self._undo.pop()
        return copy.deepcopy(self.current)

    def redo(self) -> Any | None:
        if not self._redo:
            return None
        self._undo.append(self.current)
        self.current = self._redo.pop()
        return copy.deepcopy(self.current)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)
