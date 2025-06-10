import numpy as np


class DeclineThenFlat:
    def __init__(self, start_epsilon, step_length, decay="exp", min_epsilon=0.01):
        if min_epsilon < 0 or min_epsilon > 1 or step_length < 0:
            raise ValueError("min_epsilon must be between 0 and 1")

        self.start_epsilon = max(start_epsilon, min_epsilon)
        self.min_epsilon = min_epsilon
        self.step = step_length
        self.decay = decay
        if self.decay == "linear":
            self.decay_delta = (start_epsilon - min_epsilon) / step_length
        elif self.decay == "exp":
            self.decay_delta = np.log(start_epsilon / min_epsilon) / step_length
        else:
            raise NotImplementedError(f"There is no decline way called {self.decay}")

    def eval(self, current_step):
        if self.decay == "linear":
            return max(self.min_epsilon, self.start_epsilon - self.decay_delta * current_step)
        elif self.decay == "exp":
            epsilon = np.exp(np.log(self.start_epsilon) - self.decay_delta * current_step)
            return max(self.min_epsilon, epsilon)
        else:
            return self.epsilon
