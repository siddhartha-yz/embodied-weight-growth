# Sequential SmolVLA repair: bottle → bowl → both

The same offline SmolVLA-LIBERO is edited in three consecutive opportunities. Each opportunity uses one Codex SDK call to GPT-6 Astra and returns at most eight **absolute FP32 values** for existing entries in `model.action_out_proj.weight`. There is no runtime controller, task switch, gradient update, or averaging of independently edited models.

| Version | Starts from | What Astra sees |
| --- | --- | --- |
| v0 | Original checkpoint | Baseline only |
| v1 | v0 | Difficult bottle and ordinary successful bottle, with actual actions and one-weight intervention probes |
| v2 | v1 | Difficult bowl and ordinary bowl, with probes of the **v1** network; no bottle rollout results in this call |
| v3 | v2 | Both tasks together, with probes of **v2** and summaries of v0 and v1 development outcomes |

An assignment replaces the current value, including values edited in an earlier round if Astra selects them again. Unlisted parameters stay as in the immediately preceding model. v3 changes 16 entries relative to stock and was materialized as a separate checkpoint. The evaluated version loads that full checkpoint directly. Recreate it from the archived [v3 assignments](v3.json) and [verified save values](materialization.json).

## Results

| Test | v0: stock | v1: bottle | v2: then bowl | v3: then both |
| --- | --- | --- | --- | --- |
| Development bottle | Correct lift, fails placement | Lifts bowl instead | No sustained lift | No sustained lift |
| Development bowl | Hits cabinet rim, drops | Drops | Stable at step 149 | Fails |
| Fresh bowl, stable release | 1/3 | 2/3 | **3/3** | 1/3 |
| Fresh bottle, stable release | **1/3** | 0/3 | 0/3 | **1/3** |
| Originally successful examples, native success | **8/8** | **8/8** | 4/8 | 7/8 |

v2 improves placement on the development bowl and all three new bowl layouts, while losing all three ordinary bottle examples and the two-mug example. v3 restores the three ordinary bottles and the native-instruction fresh bottle, but not the development bottle or the two-mug example. Its sole fresh-bowl success is layout 15, which stock failed; it loses the fresh bowl that stock completed. Thus the final version does not improve the aggregate success rate on either fresh task. The v2 development bowl travels with the gripper and is left on the cabinet, although the simulator never flags a sustained two-finger lift.

[Paired per-case scorecard](scorecard.json) (64 rollouts) · [Development bowl video](../../../media/sequential-bowl.mp4) · [Development bottle video](../../../media/sequential-bottle.mp4) · [Fresh bowl 15 video](../../../media/sequential-heldout-bowl.mp4)

## Evaluation

- Two development cases from the preceding experiment: the crossed bottle with the same early-instruction schedule for every version, and the rim-striking shifted bowl with the original instruction.
- Eight originally successful examples: four ordinary bowl layouts, three ordinary bottle layouts, and one LIBERO-Long two-mug placement. The two ordinary examples shown to Astra are included in this retention panel.
- Six fresh layouts preselected using reset geometry only: three bowl and three bottle. No fresh episode is run before the third edit is fixed.
- Within each case the simulator state, the two camera observations, robot state, instruction schedule and random seed match. All versions use fixed weights starting at control 1 and replan every 10 controls. For target placement, success means that the native predicate remains true for 40 consecutive controls (2 seconds) after the object leaves the grasp. A transient native hit is shown separately. Native success is used for the original-task retention panel. The renderer records every scored rollout.

## Reproduce

[Recreate the checkpoint and paired evaluations](../REPRODUCE.md#sequential-repair).
