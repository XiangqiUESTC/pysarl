from .common import (
    AllToTensor,
    AtariFireReset,
    AtariFrameSkipMax,
    AtariNoopReset,
    FlagToTensor,
    PongMinimalActions,
    PongPreprocess,
    RewardToTensor,
    StateToDiscreteIndex,
    StateToOneHot,
    StateToTensor,
)

REGISTRY = {
    "state_to_tensor": StateToTensor,
    "state_to_discrete_index": StateToDiscreteIndex,
    "state_to_one_hot": StateToOneHot,
    "reward_to_tensor": RewardToTensor,
    "flag_to_tensor": FlagToTensor,
    "all_to_tensor": AllToTensor,
    "atari_noop_reset": AtariNoopReset,
    "atari_fire_reset": AtariFireReset,
    "atari_frame_skip_max": AtariFrameSkipMax,
    "pong_minimal_actions": PongMinimalActions,
    "pong_preprocess": PongPreprocess,
}
