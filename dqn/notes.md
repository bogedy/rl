## General

**Remember that we're on Windows using powershell and the conda env is `torch-xpu`**

Currently the script runs efficiently. Let's try to keep it that way. 

"The behavior
policy during training was eps-greedy with eps annealed linearly from 1 to 0.1 over the first million
frames, and fixed at 0.1 thereafter. We trained for a total of 10 million frames and used a replay
memory of one million most recent frames."

For metrics indexing purposes, let's track training steps and epochs, and call an epoch 50000 steps like in the paper.

"More precisely, the agent sees and selects actions on every k
th frame instead of every
frame, and its last action is repeated on skipped frames. "

## Preprocessing

We only need this crop:
```
ROW_START, ROW_END = 40, 190   # rows to keep
COL_START, COL_END = 30, 140   # cols to keep (full width)
cropped = frame[ROW_START:ROW_END, COL_START:COL_END]
```

Cast to black and white and resize to 56x56.

Store in the memory bank as some compact dtype, cast up to bf16 when passed to the model. 

## Evaluation

• Every 10,000 steps, perform an evaluation.
• In each evaluation, run one or more evaluation episodes.
• Record the total reward obtained during evaluation.

Let's do 10 evaluation episodes...

Evlauations can be run in parallel while training continues. Use multithreading to accomlish this, either pytorch multithreading or standard python mt. We cannot use third party libraries, except for pytorch and ALE. The training should start with an immediate evaluation that runs while training simultaneously commences so that we can confirm that it's working. 

## Metrics

Here is the paper on Q tracking metric:
> Another, more stable, metric is the policy’s estimated action-value function Q, which provides an estimate of how much discounted reward the agent can obtain by following its policy from any given state. We collect a fixed set of states by running a random policy before training starts and track the average of the maximum predicted Q for these states. The two rightmost plots in figure 2 show that average predicted Q increases much more smoothly than the average total reward obtained by the agent and plotting the same metrics on the other five games produces similarly smooth curves.

Include code to tracking metrics during training so that they can be easily visualized afterward. 

Use W&B, something like this:
```
import wandb
wandb.init(project="dqn-atari")
# then in your log block:
wandb.log({"eps": eps, "avg_return": avg_ret, "loss": avg_loss, "sps": sps}, step=step)
# and eval:
wandb.log({"eval_return": eval_ret}, step=step)
```

Let's catch at the beginning that this is all set up properly. 