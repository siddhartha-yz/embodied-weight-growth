"""Apply the established bit-matched rollout score to the three localization arms."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location("paired_score",Path(__file__).resolve().parents[1]/"bp-control"/"score.py")
score=importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)
score.ARMS=("stock","auto-bp","astra-bp","astra-wsp")
if __name__=="__main__":
    score.main()
