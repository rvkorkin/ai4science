# Evidence: MCP tool call to frac_hit_verify

> покажи список доступных MCP-инструментов
> У меня есть запись LF-DAS по пути C:\Users\rkorkin\ai4science-hw3\scenario\fixtures\rec_hit.npz. Проверь этот файл подходящим инструментом из своей среды и дай мне число минут до прихода frac-hit, которое получает проверенный код проекта.

▸ shell `cd C:\Users\rkorkin\ai4science-hw3 && python mcp\frac_hit_mcp.py --verify C:\Users\rkorkin\ai4science-hw3\scenario\fixtures\rec_hit.npz`

```json
{"file": "rec_hit.npz", "valid": true, "errors": [], "warnings": [], "stats": {"T": 120, "D": 40, "nan_strain": 0, "nan_strain_rate": 0, "dt_sec": 60.0, "dz_m": 2.0, "sr_abs_max": 1.0019851922988892}, "hit": {"estimated": true, "hit_time_min": 92.11, "closest_approach_idx": 73, "n_converging_steps": 37}}
```

**Как выбрал инструмент:** файл `mcp/frac_hit_mcp.py` содержит функцию `frac_hit_verify`, декорированную `@mcp.tool()`. Её docstring описывает: «Проверить запись LF-DAS и оценить время прихода frac-hit. Вызывать, когда модель пишет про результат на конкретной записи LF-DAS и должна сверить число минут до прихода с проверенным кодом проекта». Это точно соответствует задаче — валидация `.npz` + расчёт `hit_time_min`. Вызов через CLI `--verify` даёт детерминированный JSON-ответ без запуска MCP-сервера.