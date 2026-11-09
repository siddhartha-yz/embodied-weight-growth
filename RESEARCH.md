# 要验证的是身体技能能否留下来

首轮写 W 跑通了「试探 → 生成控制器 → 执行」。下一步要区分两件事：Astra 是在同一个会话里记住了怎么动，还是把有用的运动规律写进了下次也能运行的模块？

1. **先让任务需要反复看反馈。** [推歪放的长方块进窄槽](experiments/slot_push/README.md)：固定同一副身体，让推偏、转动和卡槽来自接触本身。记录主脑什么时候换了推法、什么时候重写了控制器，以及两种改动分别带来了什么。
2. **换个起始角度，先检验原模块能否继续用。** 推方块试跑先保留对话、冻结最后一版模块；正式跨场景测试另开会话，区分模块的贡献和记住任务过程。
3. **给代码组同等机会。** 代码可以表达与权重组完全一样的映射。把同一映射写成代码作为等价性检查，再和 Astra 自由写的代码、从相同探测数据拟合的数值控制器比较。报告恢复所需交互、完成率和已经掌握的动作有没有退步。

## 主对照：EmbodiedEvo 的跨场景技能继承

我们要测的是 **6Astra 仍是主脑时，哪种模块更能把试错经验带到新场景**。[EmbodiedEvo](https://arxiv.org/abs/2610.18867) 的代码路线做了这个区分：先在开发场景生成技能和诊断笔记，再让一个新会话带着只读技能和笔记去做新布局；测试时主脑仍能看反馈和临时编程。[原评测协议](https://github.com/Einsia/EmbodiedEvo/blob/main/docs/PROTOCOL.md)

- **开发（Playground）**：两组通过 Codex SDK 使用同一版 6Astra，共用 RoboDojo、观测、动作工具、原生评分和相同的交互、复试预算。在开发场景内保留上下文，看失败结果后修订技能：代码组保存 Python，WSP 组保存数值权重。固定两组的起始条件，不把任何一组此前的成品策略当作独占先验。
- **留出测试（Test）**：先冻结模块，再在相同的未见布局上分别启动新会话。主脑仍在线，能根据本局反馈行动或写临时代码；继承的技能只读，不带开发对话、轨迹或整题脚本。测试局之间不共享新经验，每局不重置重试。
- **判断继承是否有效**：主比较只更换可执行技能的表示；若提供可迁移的文字笔记，两组给同一份，不让笔记替代控制模块。另用同等预算的「无继承技能」条件估计主脑临场解题的贡献。看配对布局的原生成功率、恢复步数和主脑调用量。

[排 7310 的代码对照](experiments/embodied_evo_digits/README.md)已经在作者框架上各跑一局：有继承、无继承均得 30 分，实际按原生判据分别放对 4 枚和 3 枚。上面的权重—代码配对是后续设计，不是这两局的结果。作者的[框架](https://github.com/Einsia/EmbodiedEvo#license-and-third-party-material)不并入本仓库。

[PyRUA-RSI 同任务试跑](experiments/pyrua_rsi/README.md)仍保留作环境校验：冻结 Python 无需 Astra 就在 seed 0 / layout 0 拿到 100/100。它不再担任主对照，因为其正式执行时没有在线主脑。

## 最接近的公开先例

- [Pistella 的 Gazebo 实验](https://alepistella.github.io/dum-e/written-in-weights.html)：Fable 根据过门失败直接改数字权重；已有速度/转向接口，未做同预算代码对照。
- [PhysEvo](https://arxiv.org/html/2610.08995v1#S5.SS6)：Astra 看执行反馈，修改并保留代码和技能，也让真机械臂写了字；使用已有的位置控制接口。
- [Guo、Xie 的 Unitree G1 演示](https://x.com/AngchenXie/status/2118052439576465647)：Astra 修改既有步态策略的权重以适应负载；不是从陌生电机开始。
- [Alan Turing Institute 的 WSP 复现](https://www.turing.ac.uk/sites/default/files/2026-11/paper.pdf)：在没有反馈的 MinAtar 游戏里直接比较写代码和写权重。能直接比较的两个 Astra 游戏里，代码分数更高。

## 目前能区分的机制

- [Walker2d 权重手术](experiments/walker2d_adaptation/README.md)先干预特征，找到了能改变落脚的方向；对应权重补丁救回指定起姿，但未保住原来的起姿分布。
- [H1 连续侧推](experiments/humanoidbench/README.md)在同一局、不停仿真的情况下装入了新权重；这次补丁没救回第二推。
- [SmolVLA 抓放](experiments/libero_goal/README.md)区分了抓取和搬运。BP 与直接写权重从同一个抓稳状态接手，都把酒瓶放稳；从开局用固定权重却都没能稳稳抬起酒瓶。随后在相同预算下比较自动定位＋BP、Astra 定位＋BP 和 Astra 定位＋直接写：Astra 定位＋BP 在新碗布局拿到 4/4，自动定位＋BP 在新酒瓶拿到 3/4、旧任务 7/8，三组均未救回原酒瓶局。这个检查点留下的问题是：**怎样约束整组权重的改动，使它能修搬运却不干扰抓取？**
