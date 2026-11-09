# Verified short action targets

Both branches start on the same previously recorded stock rollouts, at a 10-control boundary after the intended object was already held. The exact stock reset simulator state and public observations were reproduced, and every prefix action, target position, end-effector position, grasp flag and native success flag matched through the branch point. The selected branches replay stock commands except for short Y/Z changes; neither changes gripper, rotation or X commands.

| Target | 0-based branch windows | Direct action changes | First native success | 40 consecutive ungrasped native-positive controls |
| --- | --- | --- | ---: | ---: |
| Bottle | 90, 100 | Y −0.22 for 20 controls; Z +0.28 for first 10 | 103 | 146 |
| Bowl | 110 | Y −0.35 and Z +0.45 for 10 controls | 123 | 162 |

Y has lower bound −1 and Z has upper bound +1. The cutover is before the action at the stated zero-based index (e.g. t=90 means the 91st control). The bottle settles on its side within the native goal; upright placement is not part of this score.

- [Bottle branch trace](bottle/trace.json) · [final frame](bottle/stable.png)
- [Bowl branch trace](bowl/trace.json) · [final frame](bowl/stable.png)

[teachers.json](teachers.json) contains native 7-D action targets. Its source and trace paths are relative to the experiment’s `src/` directory. These branches provide training evidence for both methods; they are not scored as fixed policies. [Regenerate the inputs from the archived replay fixtures](../../REPRODUCE.md#rebuild-the-shared-inputs).
