"""Run this entry point to train the optimized model in newcode/code."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'code'))
from train_dynamic import main

if __name__=='__main__':
    main()
