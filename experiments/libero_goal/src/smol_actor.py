"""Offline, editable official SmolVLA-LIBERO actor for raw OffScreenRenderEnv observations."""
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import SmolVLANewLineProcessor  # registry setup
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.utils.constants import ACTION


_MODEL_ROOT = Path(__file__).resolve().parent / "models"


class SmolActor:
    """Two RGB cameras, 8D robot state, one instruction -> 7D native LIBERO action."""

    def __init__(self, checkpoint_dir=None, vlm_assets_dir=None, device="cuda", n_action_steps=None):
        checkpoint_dir = Path(checkpoint_dir or _MODEL_ROOT / "smolvla_libero").resolve()
        vlm_assets_dir = Path(vlm_assets_dir or _MODEL_ROOT / "smolvlm2").resolve()
        self.device = torch.device(device)
        cfg = PreTrainedConfig.from_pretrained(checkpoint_dir, local_files_only=True)
        cfg.device = str(self.device)
        cfg.vlm_model_name = str(vlm_assets_dir)
        # The policy checkpoint already has the full retained VLM. The other repository
        # is needed for tokenizer/config only, not its original multi-GB weights.
        cfg.load_vlm_weights = False
        if n_action_steps is not None:
            if not 1 <= n_action_steps <= cfg.chunk_size:
                raise ValueError(f"n_action_steps must be in [1, {cfg.chunk_size}]")
            cfg.n_action_steps = n_action_steps
        self.policy = SmolVLAPolicy(cfg)
        state = load_file(checkpoint_dir / "model.safetensors", device="cpu")
        # assign=True retains the published mix of BF16 and FP32 parameters.
        self.policy.load_state_dict(state, strict=True, assign=True)
        self.policy.to(self.device).eval().requires_grad_(False)
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint_dir,
            config_filename="policy_preprocessor.json",
            overrides={
                "tokenizer_processor": {"tokenizer_name": str(vlm_assets_dir)},
                "device_processor": {"device": str(self.device)},
            },
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint_dir,
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        self.env_processor = LiberoProcessorStep()
        self.generator = torch.Generator(device=self.device)
        self.reset(0)

    @property
    def queued_actions(self):
        return len(self.policy._queues[ACTION])

    def reset(self, seed):
        self.policy.reset()
        self.generator.manual_seed(int(seed))

    def flush_actions(self):
        """Discard queued actions at a forced replan boundary without reseeding."""
        self.policy.reset()

    @staticmethod
    def _rgb(image):
        image = np.ascontiguousarray(image)
        if image.shape not in ((256, 256, 3), (360, 360, 3)) or image.dtype != np.uint8:
            raise ValueError(
                f"expected raw 256x256 or 360x360 RGB uint8; got {image.shape}, {image.dtype}"
            )
        return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().div_(255)

    @staticmethod
    def _state(value):
        return torch.as_tensor(np.ascontiguousarray(value)).unsqueeze(0)

    @torch.inference_mode()
    def prepare(self, obs, instruction):
        batch = {
            "observation.images.image": self._rgb(obs["agentview_image"]),
            "observation.images.image2": self._rgb(obs["robot0_eye_in_hand_image"]),
            "observation.robot_state": {
                "eef": {
                    "pos": self._state(obs["robot0_eef_pos"]),
                    "quat": self._state(obs["robot0_eef_quat"]),
                },
                "gripper": {"qpos": self._state(obs["robot0_gripper_qpos"])},
            },
            "task": [instruction],
        }
        return self.preprocessor(self.env_processor.observation(batch))

    def make_noise(self, batch_size=1):
        cfg = self.policy.config
        return torch.randn(
            (batch_size, cfg.chunk_size, cfg.max_action_dim),
            dtype=torch.float32,
            device=self.device,
            generator=self.generator,
        )

    @torch.inference_mode()
    def predict_action_chunk(self, batch, noise=None, denormalize=True):
        """Probe a full chunk without consuming the execution queue."""
        if noise is None:
            noise = self.make_noise(batch["observation.state"].shape[0])
        chunk = self.policy.predict_action_chunk(batch, noise=noise)
        return self.postprocessor(chunk) if denormalize else chunk

    @torch.inference_mode()
    def act(self, obs, instruction):
        # While draining a chunk, upstream select_action ignores the batch.
        if self.queued_actions:
            action = self.policy.select_action({})
        else:
            action = self.policy.select_action(
                self.prepare(obs, instruction), noise=self.make_noise()
            )
        return self.postprocessor(action)[0].cpu().numpy().astype(np.float32, copy=False)
