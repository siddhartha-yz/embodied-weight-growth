"""The EmbodiedEvo program API over HumanoidBench position actions."""
from simulators.primitives import PrimitiveApi


class HumanoidBenchLowLevelApi(PrimitiveApi):
    def functions(self):
        return {name: getattr(self, name) for name in
                ("get_instruction", "get_observation", "step", "reset")}

    def get_instruction(self):
        return self._env.task_prompt()

    def get_observation(self):
        return self._env.public_observation()

    def step(self, action):
        return self._env.control_step(action)
