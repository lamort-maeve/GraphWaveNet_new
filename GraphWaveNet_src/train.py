"""Dynamic PCMCI Graph WaveNet training; implementation and outputs live in result/."""
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "result" / "code"
sys.path.insert(0, str(CODE))
from train_dynamic import main

if __name__ == "__main__":
    main()
