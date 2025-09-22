from dataclasses import dataclass
from typing_extensions import Literal, Union, Tuple, Optional


@dataclass
class Scheme:
    """
    Scheme是用于描述环境个参数的类，包括状态空间和动作空间的连续或离散性，
    动作的数量、维度，或者状态的数量、维度，用于指导controller的创建，
    也能用于判定环境和算法是否相容
    """

    action_space_type: Literal["discrete", "continuous", "composite"] = None
    discrete_action_size: Optional[Union[int, Tuple[int, ...]]] = (
        None  # 离散动作数量或形状
    )
    continuous_action_shape: Optional[Tuple[int, ...]] = None  # 连续动作形状

    state_space_type: Literal["discrete", "continuous", "composite"] = None
    discrete_state_size: Optional[Union[int, Tuple[int, ...]]] = (
        None  # 离散状态数量或形状
    )
    continuous_state_shape: Optional[Tuple[int, ...]] = None  # 连续状态形状

    max_step: int = None  # 环境的最大步长

    def __post_init__(self):
        # 校验动作空间类型值是否为3个中的1个
        if self.action_space_type and self.action_space_type not in (
            "discrete",
            "composite",
            "continuous",
        ):
            raise ValueError(
                'action_space_type must be one of "discrete" "continuous" or "composite"!'
                f"Now it is {self.action_space_type}"
            )
        # 校验状态空间类型值是否为3个中的1个
        if self.state_space_type and self.state_space_type not in (
            "discrete",
            "composite",
            "continuous",
        ):
            raise ValueError(
                'state_space_type must be one of "discrete" "continuous" or "composite"!'
                f"Now it is {self.state_space_type}"
            )

        # todo 一些其他的检验
