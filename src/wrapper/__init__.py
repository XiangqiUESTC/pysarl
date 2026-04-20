from .common import AllToTensor, FlagToTensor, RewardToTensor, StateToOneHot, StateToTensor

REGISTRY = {
    "state_to_tensor": StateToTensor,
    "state_to_one_hot": StateToOneHot,
    "reward_to_tensor": RewardToTensor,
    "flag_to_tensor": FlagToTensor,
    "all_to_tensor": AllToTensor,
}
