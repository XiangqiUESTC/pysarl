from .common import StateToTensor, RewardToTensor, AllToTensor, FlagToTensor

REGISTRY = {
    "state_to_tensor": StateToTensor,
    "reward_to_tensor": RewardToTensor,
    "flag_to_tensor": FlagToTensor,
    "all_to_tensor": AllToTensor,
}