import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdsd.evidence.runner import main as evidence_main


if __name__ == "__main__":
    args = sys.argv[1:] or ["--artifacts", "artifacts", "--with-pytest"]
    raise SystemExit(evidence_main(args))
