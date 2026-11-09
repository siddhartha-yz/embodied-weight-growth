# 让 Astra 自己学会走路

和推方块不同，这里没有「把手送到目标点」的接口。Astra 只能给六个关节发力，或写一个每 20 ms 看传感器、发出六路力矩的模块。身体受重力、摩擦和落脚冲击；不提供站立控制器、步态、参考轨迹或预训练策略。

- 身体使用 Gymnasium 1.2.1 随附的 Walker2d-v5 模型（`walker2d.xml`，保留 MIT 许可）。两组用同一副身体和同一初始姿态，每次摔倒可以复位，模型上下文和已花预算保留。
- 权重组直接写稀疏数值 MLP；代码组写受限 Python。两组看到同样的关节角、速度、机身姿态、触地信号和时间特征。模型在动作段之间观察与决策，模块在段内做闭环控制。
- 每组上限 36 次模型决策、24 段运动、1,600 个控制步；每段最多 250 步。判定要求至少运行 2 秒、身体向前至少 1 m、至少四次有效的交替落脚且仍直立。一步要求摆动脚离地至少 0.06 秒、落点向前至少 45 mm，且摆动期间另一只脚曾支撑。滑动、跌倒时冲出去的距离不算完成。
- 首轮看实际观察是否促成模块更新。把第一版和试跑中表现最好的后续版冻结，在同一初始状态和新初始倾角下各跑相同的时长，不再让 Astra 参与。这样能区分模块本身的改变和上下文里记住了怎么做。

## 首轮结果（seed 271）

在线试跑中，代码 v8 使机器人走了 1.235 m，累计 5 次交替落脚，5.20 秒时仍直立。权重组有合格的单侧落脚，但未形成连续的左右交替步态。

在课题组服务器上冻结控制器、从初始状态重放（至多 5 秒，无 Astra 参与）：

| 控制器 | 初始俯仰 0.02 rad | 新初始俯仰 0.08 rad |
| --- | --- | --- |
| 权重 v1 | 0.12 秒摔倒，0 步 | 0.14 秒摔倒，0 步 |
| 权重 v8 | 1.64 秒摔倒，1 步 | 1.42 秒摔倒，1 步 |
| 代码 v1 | 0.16 秒摔倒，0 步 | 0.16 秒摔倒，0 步 |
| 代码 v8 | 5 秒未倒，2.71 m / 11 步 | 5 秒未倒，2.87 m / 11 步 |

这里的步数是满足前述规则的交替落脚计数。代码 v8 按时钟相位区分支撑腿和摆动腿的控制；冻结结果表明这个版本可以独立起步。比较仅覆盖一副身体、一次生成过程和一个额外起始倾角。

[后续：把代码 v8 编译进同一个小网络](COMPILATION.md)；冻结权重在两个起始倾角都走满了 5 秒。

## 运行与查看

使用仓库首页的 MuJoCo / Codex SDK 依赖。在本机已登录 Codex 的情况下：

```bash
run=runs/walk-pilot
python experiments/walk/run_walk.py --arm weights --out "$run/weights"
python experiments/walk/run_walk.py --arm code --out "$run/code"
python experiments/walk/audit_walk.py "$run" --out "$run/audit.json" --verify-replay --ablate
python experiments/walk/render_walk_replay.py --out "$run/report" \
  --run "权重=$run/weights/train" --run "代码=$run/code/train"
```

如用课题组服务器运行物理仿真，给前两条及审计命令附上 `--ssh-host`、`--python`、`--remote-server`，已有运行时可指定 `--remote-pythonpath`。服务器需要 `experiments/walk` 和同级的 `w_writer`。模型会话仍在客户端运行。

## 已记录的试跑

[审计、模块和冻结重放](../../results/271-walk-v1/) · [过程总览](../../results/271-walk-v1/overview.png)。
