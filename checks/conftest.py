import sys
from pathlib import Path

# The functions under property-test (preprocess.py) live in src/ inside this
# submission repo, so the verifier runs standalone (no external project needed).
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
