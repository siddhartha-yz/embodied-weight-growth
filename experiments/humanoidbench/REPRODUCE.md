# H1 源码使用方式

这些是生成本页证据所用的实验源文件和小型结果。需要自行准备 Linux、Python、MuJoCo EGL、[固定版本 HumanoidBench](https://github.com/carlosferrazza/humanoid-bench/tree/cb1189039151c8aadaaa987b442da54383c87fab)、NumPy、PyTorch、Gymnasium、Pillow、imageio/FFmpeg，以及兼容的 [EmbodiedEvo](https://github.com/Einsia/EmbodiedEvo) 会话模块。生成新候选另需有权限的 Codex SDK/6Astra；复测冻结候选不需要模型调用。没有复刻实验机、会话服务或上传大型运行记录。

## 已训练 H1：复测冻结补丁

从 [DR.Q 指定版本](https://huggingface.co/dmux/DR.Q/tree/d31f529b9eef3999f353498a9cb66c35eb8021b2/DRQ%2BHBench-h1-walk-v0%2B0) 取得对应任务的 `encoder.pt` 和 `policy.pt`。在本目录运行：

```sh
python trained-push/source/actor_source.py \
  --encoder /path/to/encoder.pt --policy /path/to/policy.pt \
  --out trained-push/drq_checkpoint/actor_bundle.npz

export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/trained-push/source:/path/to/EmbodiedEvo:${PYTHONPATH:-}"
python trained-push/source/actor_smoke.py trained-push/drq_checkpoint/actor_bundle.npz
python matched/source/matched.py --prior trained-push --run /path/to/fresh-run
```

`trained-push/run/heldout.json` 是从原评测提取的三组逐场景指标，以及每个原策略侧推场景的三个精确状态/动作校验点；它不是全量轨迹。`matched.py` 会重新执行全部原生控制步。完整新评测可用 `trained-push/source/compare.py`，输入 [冻结计划](trained-push/calibrated/plan.json) 和自行配置的 SDK 桥。新试跑输出不会写入公开证据目录。

单推在线入口是 `live/source/run_live.py`（显式提供 `--actor`、`--plan`、`--previous trained-push/run/heldout.json`、新的 `--run` 和本地授权 `--token-file`）。双推入口是 `two-push/source/two_push.py --prior trained-push --run <新目录> --token-file <本地凭证文件>`。各阶段的 `source/brain_bridge.py` 保留原实验请求格式；凭证由读者配置，不在源码或公开结果中。重跑会再次调用模型，返回的补丁不保证相同；归档的数值差分供固定比对。

## 从零：在 EmbodiedEvo 会话中加载适配器

分别使用 [代码工作区](zero/source/code/task/) 和 [WSP 工作区](zero/source/wsp/task/)。把代码组的 `humanoidbench_adapter.py` 与 `humanoidbench_api.py` 放到上游 `simulators/`，从上游根目录执行代码组的 `register_backend.py` 注册 H1 后端。把对应工作区的 `task/`、`sdk_remote_agent.py` 和 `launch_author.py` 放在同一个实验入口目录；工作区的 `scripts/env.py` 由上游同版本工作区提供。WSP 阶段在独立副本中另把 `humanoidbench_weight_task.py`、`weight_policy.py` 放入 `simulators/`，令 H1 后端构造 `HumanoidBenchWeightTask`；基础任务和相机接口仍来自同一个代码组适配器。

再用对应 `launch_author.py explore` 指定新的 `--run`、原生解释器 `--sim-env`、自己的 SDK 桥 `--brain` 与凭证文件。代码组冻结技能和权重组 18 次安装的数值都在 [zero](zero/) 下。没有把 20 秒原地站立计作步行，也没有执行独立 Test 场景。
