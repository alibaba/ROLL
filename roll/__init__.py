# set RAY_DEDUP_LOGS=0 before importing ray
import os
os.environ["RAY_DEDUP_LOGS"] = os.getenv("RAY_DEDUP_LOGS", "1")

# Enable deterministic mode if DETERMINISTIC_MODE environment variable is set
if os.getenv("DETERMINISTIC_MODE", "0") == "1":
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    print("Deterministic mode enabled")
