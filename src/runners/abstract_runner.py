from abc import ABC
from abc import abstractmethod
from envs import REGISTRY as env_REGISTRY
from controllers import REGISTRY as controller_REGISTRY


class AbstractRunner(ABC):
    """
        为所有runner定义的抽象类，runner类是控制agent和environment进行交互，并通过buffer收集交互数据的类，
        所有的runner都应该实现AbstractRunner类中的抽象方法，
        不同类别的runner为不同的强化学习算法提供了不同粒度的检查是否参数更新的频率
    """
    def __init__(self, args, logger):
        """
            args: 本次算法运行的所有参数
            logger: 日志实例对象

            初始化方法，初始化env，controller属性为None，以及一些其他的必要数据，记录参数和logger
        """
        self.env = None
        self.controller = None
        self.scheme = None

        self.args = args
        self.logger = logger

        # 所有已经跑过的episode的所有步数的累加，上限由参数t_max决定
        self.t_env = 0
        # 在一个episode中的交互步数
        self.t = 0

        # 装配env和controller
        self.setup()

    @abstractmethod
    def run(self):
        pass

    def setup(self):
        """
            在init初始化方法之后立即调用，用于初始化环境env和controller
        """
        self.env = env_REGISTRY[self.args.env](self.args.env_args)

        # 获取环境scheme，用于初始化controller
        scheme = self.env.get_scheme()

        self.scheme = scheme

        # TODO 在这里要检查算法和环境是否相匹配

        # 使用scheme初始化controller
        self.controller = controller_REGISTRY[self.args.controller](self.args, scheme)

    # 重置一个episode中的环境
    def reset(self):
        self.env.reset()
        self.t = 0
