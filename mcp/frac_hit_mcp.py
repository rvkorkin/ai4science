"""frac_hit_mcp — MCP-инструмент под потребность проекта LF-DAS.

Один инструмент: `frac_hit_verify` берёт запись LF-DAS в формате `.npz` и
возвращает JSON-вердикт --- валиден ли файл и оценку времени прихода frac-hit
в минутах по кинематике сходящегося клина. Инструмент нужен, когда модель
пишет раздел про результаты и должна сверить число (минуты до прихода) с
тем, что даст наш собственный проверенный код, а не с тем, что она «помнит».

Регистрация в харнессе AI4Science (stdio MCP-сервер):
    command: <python-из-.venv> mcp/frac_hit_mcp.py

Проверка без агента:
    python mcp/frac_hit_mcp.py --selftest        # код 0 при успехе
    python mcp/frac_hit_mcp.py --verify <.npz>   # прямой вызов из CLI
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tool_core import frac_hit_verify as _verify  # noqa: E402

try:                       # пакет mcp 1.x в окружении учебного агента
    from mcp.server.fastmcp import FastMCP
except ImportError:        # без пакета mcp работают --selftest и --verify
    class FastMCP:         # type: ignore[no-redef]
        def __init__(self, name: str) -> None:
            self.name = name

        def tool(self):
            return lambda f: f

        def run(self) -> None:
            sys.exit(
                "нет пакета mcp: запускайте интерпретатором из окружения учебного агента")


if sys.platform == "win32":  # протокол MCP --- UTF-8
    for _s in (sys.stdin, sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

mcp = FastMCP("frac-hit")


@mcp.tool()
def frac_hit_verify(npz_path: str, max_errors: int = 10) -> str:
    """Проверить запись LF-DAS и оценить время прихода frac-hit.

    Вызывать, когда модель пишет про результат на конкретной записи LF-DAS и
    должна сверить число минут до прихода с проверенным кодом проекта, а не
    со своей памятью. Также вызывать, когда нужно убедиться в валидности
    файла `.npz` (формы, конечность значений, монотонность глубины, dt>0).

    `npz_path` --- путь к файлу (ключи: `strain`, `strain_rate`, `depth`,
    `dt_sec`). `max_errors` --- сколько ошибок валидатора оставить в ответе.

    Возвращает JSON-строку с полями:
      `file`, `valid` (bool), `errors` (список), `warnings` (список),
      `stats` (T, D, dt_sec и т.п.), `hit`
      (`estimated`, `hit_time_min`, `closest_approach_idx`,
      `n_converging_steps`). Файл только читается, ничего не меняется.
    """
    v = _verify(npz_path)
    v["errors"] = list(v.get("errors", []))[:max_errors]
    return json.dumps(v, ensure_ascii=False)


def _selftest() -> None:
    """Проверка без агента: на детерминированной фикстуре инструмент принимает
    валидный файл и оценивает время прихода около 92 мин."""
    sys.path.insert(0, str(HERE.parent / "scenario"))
    import make_fixtures  # noqa: E402
    make_fixtures.build()
    fx = HERE.parent / "scenario" / "fixtures" / "rec_hit.npz"
    assert fx.is_file(), f"нет фикстуры {fx}"
    v = json.loads(frac_hit_verify(str(fx)))
    assert v["valid"] is True, v
    assert v["hit"]["estimated"] is True, v
    t = v["hit"]["hit_time_min"]
    assert 80.0 < t < 100.0, f"ожидали hit_time около 90, получили {t}"
    # битый файл должен быть отклонён
    bad = HERE.parent / "scenario" / "fixtures" / "bad_dt.npz"
    vb = json.loads(frac_hit_verify(str(bad)))
    assert vb["valid"] is False and vb["errors"], vb
    print(f"frac_hit_verify: проверки пройдены; "
          f"hit_time_min={t} (эталон ~90), errors={vb['errors'][:1]}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif "--verify" in sys.argv:
        path = sys.argv[sys.argv.index("--verify") + 1]
        print(frac_hit_verify(path))
    else:
        mcp.run()
