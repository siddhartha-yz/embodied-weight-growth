"""Register the single HumanoidBench task in the pinned EmbodiedEvo stage."""
from pathlib import Path


def patch(file, old, new):
    path = Path(file)
    original = path.read_text()
    if original.count(old) != 1:
        raise RuntimeError(f"Upstream interface drift: {file}")
    path.write_text(original.replace(old, new))


patch("simulators/primitives.py", '    "robodojo": "RoboDojoLowLevelApi",',
      '    "robodojo": "RoboDojoLowLevelApi",\n    "humanoidbench": "HumanoidBenchLowLevelApi",')
patch("simulators/primitives.py", '    raise ValueError(f"unknown simulator backend {backend!r}")',
      '    if backend == "humanoidbench":\n'
      '        from simulators.humanoidbench_api import HumanoidBenchLowLevelApi\n'
      '        return HumanoidBenchLowLevelApi\n'
      '    raise ValueError(f"unknown simulator backend {backend!r}")')
patch("simulators/backend.py", "    else:\n        raise ValueError",
      '    elif backend == "humanoidbench":\n'
      '        from simulators.humanoidbench_adapter import HumanoidBenchTask\n'
      '        environment = HumanoidBenchTask(cfg.simulator.env_task)\n'
      '    else:\n        raise ValueError')
patch("orchestrator/session_runner.py", '        "robodojo",\n    }:', '        "robodojo",\n        "humanoidbench",\n    }:')
