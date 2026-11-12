"""Download all 446 public pairs without running calibration or test scoring."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from water_mapping.data import acquire_pairs, load_splits

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    manifest = acquire_pairs(root, load_splits(root))
    print(f'Verified {len(manifest)} TIFFs, {sum(x["bytes"] for x in manifest):,} bytes.')
