import enum
import os


RAY_NAMESPACE = "roll"
STORAGE_NAME = "SHARED_STORAGE_ACTOR"
GENERATE_SCHEDULER_NAME = "GENERATE_SCHEDULER_ACTOR"
REWARD_SCHEDULER_NAME = "REWARD_SCHEDULER_ACTOR"

BARRIER_NAME = "BARRIER_ACTOR_NAME"

CHECKPOINT_MANAGER_NAME = "CHECKPOINT_MANAGER_ACTOR"

SCHEDULER_NAME = "scheduler.pt"
OPTIMIZER_NAME = "optimizer.pt"
DIST_OPTIMIZER_DIR = "dist_optimizer"
RNG_STATE_DIR = "rng_state"

CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "roll")

IGNORE_INDEX = -100


class GenerateStopReason(enum.Enum):
    FINISH = enum.auto()
    ABORT = enum.auto()
    MAX_LENGTH = enum.auto()
    
    
class EpisodeStopReason(enum.Enum):
    FINISH = "finish"   
    MAX_LENGTH = "max_length"         
    MAX_STEPS = "max_steps" 
    ABORT = "abort"     
    ENV_RESET_FAILED = "env_reset_failed" 
    SANDBOX_INIT_FAILED = "sandbox_init_failed" 
    ENV_TIMEOUT = "env_timeout"   
    LLM_GENERATE_FAILED = "llm_generate_failed" 
    UNKNOWN = "unknown"
    NO_SYSTEM_PROMPT = "no_system_prompt"