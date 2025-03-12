from envs import REGISTRY as env_REGISTRY


class DefaultRunner:
    def __init__(self, args, logger):
        self.scheme = None
        self.controller = None

        self.args = args
        self.logger = logger

        self.env = env_REGISTRY[self.args.env](self.args.env_args)

        # the step for all episode, up limit is set by argument t_max
        self.t_env = 0
        # the step in one episode
        self.t = 0

    def setup(self, scheme, controller):
        self.controller = controller
        self.scheme = scheme

    # 这个函数会跑完一个完整的episode，收集整个episode的交互数据
    def run(self, test_mode=False):
        self.reset()
        terminated = self.env.terminated

        while not terminated:
            # 选择动作？
            state = self.env.get_state()
            action = self.controller.select_action(state, self.t_env, self.t, test_mode=test_mode)

            # 执行动作
            state, reward, terminated, info = self.env.step(action[0])

            print(reward)
            print(state)
            print(terminated)
            # 把交互数据放入replay buffer中

            self.t += 1

        self.t_env += self.t


        pass

    # 在跑一个episode之前做一些准备
    def reset(self):
        # set step of the episode to 0
        self.t = 0
        # reset the environment
        self.env.reset()
