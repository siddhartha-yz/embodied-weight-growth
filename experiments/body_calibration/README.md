# 身体接口 pilot：Astra 先试动作，再写权重

同一副两连杆机械臂，先不给任务，让 GPT‑6 Astra 通过 Codex SDK 观察两条初始探测，再设计 2–4 条探测。它只提交一小组数值权重。权重冻结后，同一个跟踪器从新起点执行两次到点和一次描圈。Astra 不参与评测，也拿不到目标或 MuJoCo 内部参数。

试跑使用本机已登录的 Codex SDK、远程 MuJoCo。没有外部数据集。

## 这个 pilot 比什么

- 两种身体：固定的未知通道耦合；相同耦合加有记忆的换向回差和滞后。两种条件共用一个种子。
- 同一种身体内，线性数值校准、自动选阶数的数值校准、Astra 直接写权重三组拿到完全相同的探测记录。后两者可用相同的两步历史形式。
- 模块预测关节加速度，公共跟踪器据此反求电机指令；适配器文件仅含数字。只改适配器，不改物理、轨迹目标或跟踪器。
- 传感输入是关节角、关节速度和末端位置；Astra 收到的是前两者的压缩轨迹与由此计算的加速度，不涉及图像感知。几何规划由公共跟踪器负责；此次不测任务级规划，也不比较 Astra 独立写端到端控制代码。
- 主指标是运动阶段末端轨迹 RMSE；另外记录最后 0.3 s 平均末端误差、命令饱和率和阈值成功率。一个种子是闭环 pilot，不估计总体优势。

## 运行入口

物理评测用 Python 3.11、NumPy 1.26.0 和 MuJoCo 3.8.1；重新调用 Astra 另需 `openai-codex==0.160.0` 和已登录的 Codex。无需调用模型即可重新评测公开的探测记录和冻结适配器：

```bash
p=experiments/body_calibration
r=results/271-body-calibration
python "$p/eval_pilot.py" --seed 271 --condition backlash \
    --records "$r/backlash-final/records.json" --adapter "$r/backlash-final/adapter.json"
```

重新试探与生成：`python experiments/body_calibration/run_brain.py --condition backlash --seed 271 --output runs/body-backlash`。默认本机运行 MuJoCo；也可用 `--ssh-host`、`--remote-python`、`--remote-server` 指定已经配置的仿真服务。

## 首轮结果（2026-11-08，身体种子 271）

两种身体分别校准。每种各 6 条探测轨迹、240 个仿真步（4.8 秒物理时间），再用冻结接口跑同样三个未见动作。下表为三条动作的平均运动阶段末端 RMSE，越低越好。

| 身体 | 线性数值校准 | 自动选型数值校准 | Astra 写权重 |
| --- | ---: | ---: | ---: |
| 线性耦合 | 4.15 mm（3/3） | 4.11 mm（3/3） | 4.85 mm（3/3） |
| 带换向回差 | 19.22 mm（2/3） | 10.39 mm（2/3） | 3.61 mm（3/3） |

带回差时，自动数值校准的探测加速度拟合误差更低（1.99 对 3.22 rad/s²），但未见任务的闭环跟踪反而更差。Astra 的模型保留两个历史项，没有加入静态死区和位置偏置。事后诊断用同一批探测数据重新拟合 Astra 选的非零项，不使用任务分数，闭环 RMSE 为 8.00 mm（3/3）。这说明在这副身体上，选中的结构有帮助，但不足以复现全部优势。此次没有跨种子重复，也没有让 Astra 另写完整 code-as-policy 对照。

[冻结适配器、探测记录和评测结果](../../results/271-body-calibration/) · [实跑轨迹叠图](../../results/271-body-calibration/trajectory-comparison.png)。`refit_diagnostic.py` 按与评测相同的四个参数重跑事后诊断。
