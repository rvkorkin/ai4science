"""MCP-сервер, добавляющий в среду один инструмент: frac_hit_verify.

Регистрация в харнессе AI4Science (Goose) как stdio MCP-сервер:
    command: <python-из-.venv> mcp/frac_hit_mcp.py

Инструмент самоописателен: модель должна уметь вызвать его по одному docstring,
без дополнительного prompt-инжиниринга (требование ДЗ-3).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from tool_core import frac_hit_verify as _verify  # noqa: E402

mcp = FastMCP("frac-hit-verifier")


@mcp.tool()
def frac_hit_verify(npz_path: str) -> str:
    """Проверить запись LF-DAS и оценить время прихода frac-hit.

    На вход — путь к файлу DAS `.npz` (ключи: strain, strain_rate, depth, dt_sec).
    Инструмент: (1) валидирует схему и физические инварианты (формы, конечность
    значений, dt>0, монотонность глубины); (2) если данные валидны — оценивает
    время прихода frac-hit в минутах по кинематике клина (схлопывание FWHM).

    Возвращает JSON-строку:
      {"file", "valid": bool, "errors": [...], "warnings": [...],
       "stats": {"T", "D", "dt_sec", ...},
       "hit": {"estimated": bool, "hit_time_min", "closest_approach_idx",
               "n_converging_steps"}}

    Используй для проверки качества данных и ранней оценки времени frac-hit по
    одной записи. npz_path — абсолютный путь или относительно рабочей папки.
    """
    return json.dumps(_verify(npz_path), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
