# PhysEvo 同题重建：用权重控制抓取与放置

在原生 RoboDojo-Sim 的 **Arrange Largest Number** 里，双臂要把散落的数字抓起来，从左到右排成最大的数。沿用本地已有的 ARX‑X5、seed 0、layout 0，评分仍由原生任务执行，不让它给控制策略传答案。该任务对应 [PhysEvo 表 2 的 Largest number](https://arxiv.org/html/2610.08995v1#S5.SS3)；本实验重建的是同题控制路线，未使用作者未公开的演化后 harness 或布局种子。

## 哪些是权重，哪些是固定接口

- **任务选择**：Astra 看头部 RGB 图，读出数字并给出数字→圆垫和手臂的分配。固定观察工具只按可见颜色给物体标记，用相机标定和公开桌面/物体高度换算目标；不读取物体真值位姿。
- **动作与恢复**：Astra 只提交稀疏数字矩阵。一个分阶段网络读取源/目标、推算的抓取中心、实测夹爪开度及上一动作结果；输出移动、夹爪、完成或重新观察，以及位置、偏航角、夹爪命令和下一阶段。运行器不提供接近、闭合、提起、放置或失败转移的序列，也不允许 Astra 直接下发位姿或关节命令。
- **执行器**：固定几何把俯抓位姿变成 ARX 的 link6 目标，用已安装的 RoboDojo worker 和通用位姿规划器执行。与 PhysEvo 后期演化出的直接关节目标不同，这是本轮固定的执行接口。
- **反馈**：每步记录网络输入、输出和动作回执。转移结束后 Astra 看实际图像判断物体是否落在目标垫；需要调整时只能交新权重。原生评分只在外部记录，不进模型上下文。

## 运行

需已有 RoboDojo/Auriga worker、布局 0 资产、可用 GPU，以及能导入 `openai_codex` 的 Python 环境和已登录的 Codex。将示例中的 GPU 编号换成已确认空闲的编号。生成报告另需 Pillow 和本机中文字体。此程序不安装依赖或下载任务资产。

```bash
python3 -m unittest discover -s experiments/physevo_wsp
python3 experiments/physevo_wsp/run_digits.py --host robot-sim --gpu 0 --out runs/physevo-wsp-digits
python3 experiments/physevo_wsp/report.py runs/physevo-wsp-digits
```

主脑在客户端运行，原生 simulator 通过 SSH 运行于服务器。输出 `semantic_plan.json`、各版 `policy-v*.json`、带截图的转移目录、逐步 `events.jsonl` 和最终 `native-score.json`。

## 首轮：8720 未排完

布局 0 的一次有效试跑，原生评分 **5/100，未通过**，用了 **486/1050 步**。画面确认 8、7 落在目标垫；视觉检查不判定朝向、中心误差或夹爪状态，不能代替原生评分。8 用 v1 一次放上；7 先遇到搬运路径不可行，v2 动作走完却没有搬到位，v3 加入夹爪开度判断并改了抓取高度后才放上。复用 v3 搬 2 时跟踪失败、未松爪；抓 0 时没有规划出路径。7 已耗完本轮两次改权重的总配额，因此后两次失败没有触发修订。

本轮权重只控制单枚数字的转移，没有单独的双臂归位子目标，而归位是原生通关条件之一。下一轮若要测能否通关，应把归位纳入同一个权重接口，同时让失败后的修订不集中在一个数字上。要比较代码和权重，两组需在同一执行器、观察和预算下重跑。

原生评分已返回并落盘；随后退出仿真时关闭 RPC 超时。已确认远端 worker 停止、GPU 已释放，运行日志仍保留。

首轮没有启用录像导出，但 worker 保留了动作中的原始相机帧。报告内的 `replay.mp4` 取自这批帧，按动作日志对齐仿真时间，并在每次结果处短暂停留。重做可用 `render_video.py RUN SOURCE`；`SOURCE` 放从仿真运行目录导出的 `observations/raw-*/cam_head.png`，并提供按远端文件修改时间排序的 `source-order.json`（每项为 `[mtime_ns, raw_observation_id]`；原顺序也保存在报告目录的 `replay-timeline.json`）。

## 代码路线：按论文重建任务代理与修订代理

同一个 RoboDojo 任务与本地布局。任务代理看图、分配数字和垫子，并检查每次放置的结果；另一个 6Astra 会话根据动作回执和相机画面写 Python 工具及技能说明，在本局继承并改写代码。工具在无网络、无用户目录的 `bwrap` 沙箱里执行，只能通过验证过的指令请求机器人动作；可请求绝对关节目标，也可用通用位姿规划。代码须负责抓、搬、释放、恢复以及最后双臂归位。原生评分不进入模型上下文。和权重试跑共用同一原生任务、布局和 1050 步上限，但这里增添了关节接口、更多修订额度和归位，因此两轮不构成只更换技能表示的严格对照。

```bash
python3 experiments/physevo_wsp/run_code.py --host robot-sim --gpu 0 --out runs/physevo-code-digits
python3 experiments/physevo_wsp/report.py runs/physevo-code-digits
```

需要本机可运行 `bwrap`，服务器使用已安装的 FFmpeg。输出包含 `code-v*.py`、`skill-v*.md`、归位前后实测状态和从开场启用的原生相机录像。此单布局试跑重建论文的代码修订环节；论文五个留出布局和演化后部署文件未公开。

### 单布局实测：两手抓住数字，但没有搬到目标

代码版在 layout 0 得 **0/100，未通过**（174/1050 原生步，5 个代码版本）。左手夹起 8，但转腕和搬运都出现路径不可行；改版试图接续抓持，但当次运行在重试时清空了代码状态。之后腕部图确认左手仍握着 8，新版不再将其误作 7 搬运。右手夹起 2 后也找不到搬运路径，最后两只手都没腾空。此次运行有 42 次不推进物理的位姿预览，没有执行直接关节目标，和论文演化后由策略选择关节构型的流程有实质差别。

当前运行器已加入显式的同物体重试状态迁移。用本次保留的输入重放 v2 第一个决策时，给旧状态会输出搬运，不给则要求重新观察；该修复尚未重跑物理验证。若要继续，优先让任务代理有权暂停排程并处理手里卡住的物体，再去派发新的数字。

## 已记录的试跑

[权重组原生评分](../../results/physevo-digits/weights/native-score.json) · [代码组原生评分](../../results/physevo-digits/code/native-score.json) · [权重组画面总览](../../results/physevo-digits/weights/overview.png) · [代码组画面总览](../../results/physevo-digits/code/overview.png)。
