# 把歪放的方块推入窄槽

两组 GPT‑6 Astra 都从陌生的两个电机通道开始，在同一个 MuJoCo 接触任务里主动试探。模型每次决定试哪个电机、把推杆移向哪里，或是否重写模块；模拟器只执行和返回观测，不提供路线、控制器或抓推原语。

## 试跑协议

- 同一个两连杆机械臂、同一个固定电机映射（seed 271），不会在途中改变。滑块通过真实仿真接触旋转或平移；桌面阻力用平面关节的摩擦与阻尼近似。
- 初始角度先是 +30°，再换成 −30°。其余初始条件与目标不变。方块长 38 mm、宽 20 mm，槽净宽 29 mm。成功需要四个角都进目标区、长轴与槽方向相差不超过 9°（模 180°），且平移速度不超过 14 mm/s。
- 每组每阶段最多 45 次模型决策、36 段运动、2,400 个 20 ms 的物理步。两组每步都看到实际俯视图、关节和推杆状态、物块状态、接触及动作结果。权重组直接提交 23→2 或带一层 tanh 的小网络；代码组提交受限 Python。二者使用相同的传感数据与目标点接口，表达能力并不相同。
- 换角度时保留各组的 Astra 会话，冻结首阶段最后一次成功安装的模块；禁用重装与直接电机指令。该阶段检验「主脑带着旧经验能否继续用原模块」，而不是单独测模块相对于上下文的贡献。

## 运行

按仓库首页配置 Python、MuJoCo 和 Codex SDK。用本机模拟器时：

```bash
python experiments/slot_push/run_slot.py --arm weights --out runs/slot-pilot/weights
python experiments/slot_push/run_slot.py --arm code --out runs/slot-pilot/code
```

用服务器模拟器时，在两条命令后附 `--ssh-host`、`--python`（服务器解释器）、`--remote-server`（服务器上的 `slot_server.py`），如需复用现成的依赖目录，再附 `--remote-pythonpath`。服务器需包含 `slot_push` 和同级的 `w_writer`（共用机械臂和受限解释器）；SDK 和模型会话仍在客户端。

每组按 `train` / `transfer` 分目录。`views/` 保存每次发给模型的实际 PNG 和文字观测；`events.jsonl` 保存它选的动作、简短解释、完整轨迹、接触和预算；`policy-v*.json` 保存每一次成功安装的模块。

## 查看每次动作

生成报告不调用 Astra。每步可展开实际输入图、接触反馈，以及下一步选择：

```bash
run=runs/slot-pilot
python experiments/slot_push/render_slot_replay.py --out "$run/report" \
  --run "权重 · 初始=$run/weights/train" --run "代码 · 初始=$run/code/train" \
  --run "权重 · 换角度=$run/weights/transfer" --run "代码 · 换角度=$run/code/transfer"
python experiments/slot_push/audit_run.py "$run" --out "$run/audit.json"
```

打开生成的 `report/index.html`。短暂触壁不等同于卡住；日志把「持续接触推杆和槽壁、方块几乎不动」单独标为 `stall`。

## 已记录的试跑

[审计与逐轮观测](../../results/271-slot-v1/) · [过程总览](../../results/271-slot-v1/overview.png)。代码和权重组均只安装了一次，换角度不改模块。

| 组别 | +30° 初始 | −30° 冻结迁移 |
|---|---:|---:|
| 权重 | 成功，18 段 / 1090 步 | 成功，6 段 / 425 步 |
| 代码 | 成功，17 段 / 1025 步 | 成功，16 段 / 1003 步 |

迁移继续使用各自原对话，不是新会话技能继承。这里的差异来自单个种子的模型交互和模块组合，不能单独归因到表示。
