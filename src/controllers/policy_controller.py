from .abstract_controller import AbstractController


class PolicyController(AbstractController):
    def __init__(self, scheme, args):
        super().__init__(scheme, args)

    def select_action(self):
        pass

    def forward(self):
        pass

    def _build_agent(self, input_shape):
        pass

    def _build_inputs(self, batch):
        pass