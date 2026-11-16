# STATUS

**Branch**: `article` · **Tag**: `article-v1` · **PDF**: `article.pdf` (4 pages main + 1 page refs/ai-usage)

## Where we are

Article **submitted** to the week-10 AI4Science-YSDA mini-conference.
Using the official template `ai4science_ysda2026.sty` (two-column, bibtex).
Language: Russian (English version pending agreement with instructor).

## Artifacts

| Path | What it is |
|------|------------|
| `article.tex`, `refs.bib` | Paper source and bibliography (15 refs, all papers/conferences with DOI) |
| `article.pdf` | Compiled paper, committed for convenience |
| `figs/` | Figures used in the paper (LF-DAS diagram, FORGE stage, two-hump P(hit)) |
| `ai4science_ysda2026.{sty,bst}`, `algorithm*.sty`, `fancyhdr.sty` | Template files, committed at root so `pdflatex ; bibtex ; pdflatex ; pdflatex` works out of the box |
| `METHOD.md` | Methodology text supporting §2–§4 |
| `LITREVIEW.md` | Expanded literature review draft |
| `AGENTS.md`, `AGENT.md` | How AI assistants are used in this project (course rubric R6) |
| `WRITING.md` | Style guide (register, abbreviations, Russian/English switch) |
| `GLOSSARY.md`, `ONTOLOGY.yaml` | Symbols and entity/relation map used in the paper |
| `openspec/REQUIREMENTS.md` | Acceptance criteria R1–R8 for the article |
| `diary.md` | Session-by-session log of decisions |
| `checks/` | Verification stack carried over from HW2: `validate_data.py`, `test_*.py`, `experiment.py` (leak-cost, 5 seeds), `CausalPreprocess.lean` (4 theorems, 0 sorry) |
| `src/preprocess.py`, `src/wedge_near.py` | Code under verification |

## How to reproduce

```powershell
# Paper
pdflatex article ; bibtex article ; pdflatex article ; pdflatex article

# Verification + leak-cost experiment
python -m pytest -q checks
```

Environment: Python 3.11, numpy 1.26, torch 2.3, scikit-learn 1.5, Lean 4 + SciLib,
MiKTeX 24.1.

## Last commit

See `git log --oneline -1` on branch `article`. Tag `article-v1` marks the submitted PDF.
