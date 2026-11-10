# H1 冻结复测与离线审计

在本目录运行以下命令。冻结复测与离线审计都无需调用 Astra。新模型提案不保证可重现；这里发布原始数值更新和评估逻辑，不含 SDK 会话与原运行环境。

## 离线核对参数与踝关节限幅

Python 3.11 或更新版本可核对所有接受的绝对系数提案，并从任一父版本链还原指定权重。普通审计只需要标准库；`--ankles` 另需 NumPy。

```sh
python3 source/audit.py
python3 source/audit.py --export adaptive/r442 /tmp/h1-r442.json
python3 source/audit.py --ankles
```

- [检查点索引](results/checkpoints.json) 把版本号映射到可直接加载的冻结权重。两个独立分支用 `fixed/r30` 与 `adaptive/r30` 区分；自选分支 r01–r450 之间不重编号。
- [原始短支线](results/initial-rounds.jsonl)、[固定慢走分支](results/fixed-rounds.jsonl)、[自选试验分支](results/adaptive-rounds.jsonl) 各用一行 JSON 表示一轮。字段 `parent` 是更新基底；`proposal.rows` 内的 `to`、`sources`、`values` 指定输出行和绝对赋值。自选分支的 `plan` 是实际执行的计划，`development` 包括执行的速度段、种子、接触和结果。字段 `plan_retries` 另记计划补交；[补交原始记录](results/plan-repair.json) 区分 r148 的无效请求与 r345 最终生效的更新。
- [冻结逐局结果](results/heldout.json) 保留原评估的完整判定、失稳前位移、合格落脚以及两次测量的初始骨盆状态。对照分析的 `upright_walk` 只计首次持续失稳前；评估器的整局字段（例如 `longest_alternating_run`）不可直接当成失稳前指标。
- `--ankles` 使用两条压缩的 r150 原始盲测轨迹（2801、2803）；先验证 r150 重算动作和记录一致，再对同一状态计算 r270 和 r450 的动作。分母是每局 1,000 个控制步，没有重新调用物理引擎。

## 冻结权重复测

先按 [固定版本 HumanoidBench 的安装说明](https://raw.githubusercontent.com/carlosferrazza/humanoid-bench/cb1189039151c8aadaaa987b442da54383c87fab/README.md) 准备 Python 3.11 原生 MuJoCo 环境（不是 MJX 执行后端）；保留上游 `h1-walk-v0` 原始资源。原实验核心版本为 NumPy 1.26.0、Gymnasium 0.29.1、MuJoCo 3.1.6。复测用的是本仓库的稀疏 JSON 策略，不下载上游预训练控制器。录制视频另需 Pillow、imageio、imageio-ffmpeg 和可用的 MuJoCo 渲染后端（无显示环境可设置 `MUJOCO_GL=egl`）。

```sh
python3 source/plant.py describe
python3 source/plant.py evaluate --weights weights/adaptive-r150.json --suite walk-test --out /tmp/h1-r150/walk
python3 source/plant.py evaluate --weights weights/adaptive-r150.json --suite stand-test --out /tmp/h1-r150/stand
```

`describe` 返回当前运行环境的特征、关节、动作范围和时间步，可和 [原实验接口](source/plant-describe.json) 核对；`evaluate` 为每个起姿输出完整状态动作轨迹和 `summary.json`。在命令末尾加 `--video` 可渲染各局。要渲染已有轨迹而不重跑物理，可先解压它：

```sh
gzip -dc results/r150-walk-2801.trace.json.gz > /tmp/h1-r150-walk-2801.json
python3 source/plant.py render --trace /tmp/h1-r150-walk-2801.json --out /tmp/h1-r150-walk-2801.mp4
```

## 复测 Astra 当时选的探针

下面的命令还原自选 r345 的冻结权重，提取该轮实际执行的计划，并只重跑开发探针。其他自选轮可用同样的版本号替换两处 ID。

```sh
python3 source/audit.py --export adaptive/r345 /tmp/h1-r345.json
python3 - <<'PY'
import json
from pathlib import Path
records = (json.loads(line) for line in Path("results/adaptive-rounds.jsonl").read_text().splitlines())
plan = next(row["plan"] for row in records if row["id"] == "adaptive/r345")
Path("/tmp/h1-r345-plan.json").write_text(json.dumps(plan))
PY
python3 source/adaptive_probe.py --weights /tmp/h1-r345.json --plan /tmp/h1-r345-plan.json --out /tmp/h1-r345-probe
```
