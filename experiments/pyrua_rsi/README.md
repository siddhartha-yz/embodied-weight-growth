# PyRUA-RSI：冻结代码参考

用作者公开的 [PyRUA-RSI](https://github.com/DAGroup-PKU/PyRUA-Lean/tree/main/integrations/xpolicylab/PyRUA_RSI) 运行 RoboDojo「排列最大数字」。先复查已用过的 seed 0 / layout 0，原生上限为 1050 步。程序仍看 RGB 与机器人状态并在一局内处理反馈；作者发布的本地回放端点送出固定 Python 程序，评测时不调用大模型。

## 首次本地试跑结果

冻结程序排出 **8720**，RoboDojo 原生判定 **通过，100/100**。单回合用 **470/1050** 原生动作步；三路原生相机各 471 帧、25 fps。策略记录只有一条本地回放请求、一次 Python 执行、零模型 token。运行 ID：`pyrua-rsi-digits-20261108T123457Z`。本页发布的是原生裁判记录、策略汇总和头部相机录像。保留此结果用于校验同一任务的环境；[主对照](../../RESEARCH.md)改用 EmbodiedEvo 的跨场景技能继承协议。

## 复跑

在仓库根目录，使用课题组服务器上已装好的 RoboDojo、PyRUA-Lean、SAM3 和任务资产。把示例的 GPU 换成已确认空闲的卡。

```bash
id="pyrua-rsi-digits-$(date -u +%Y%m%dT%H%M%SZ)"
ssh -T -o BatchMode=yes robot-sim bash -s -- "$id" 6 < experiments/pyrua_rsi/run_layout0.sh
```

脚本从本地已有的 PyRUA-Lean Git 对象提取固定版本的原策略和运行时，在独立目录调用上游 `eval.sh`。不改共享环境、不下载依赖。输出位于服务器 `$HOME/data/robodojo-work/archive/pyrua/$id`；`protocol.json` 记录版本，`bench/eval_result/` 下保存原生裁判结果及相机录像，`policy-records/` 下保存工具运行记录。

[原生裁判记录和首轮摘要](../../results/pyrua-rsi-digits/) · [结束画面](../../results/pyrua-rsi-digits/final.png)。
