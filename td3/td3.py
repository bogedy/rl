import argparse
import os
import copy
import time
import random
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from tqdm import tqdm
import wandb


# 
# CLI
# 
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="HalfCheetah-v5",
                    help="MuJoCo environment name (e.g. HalfCheetah-v5, Ant-v5)")
args = parser.parse_args()

# 
# hyperparams (TD3 defaults)
# 
ENV_NAME = args.env
SEED = 42

GAMMA = 0.99
TAU = 0.005 # polyak coefficient for smooth target updates
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
HIDDEN = (256, 256) # paper uses (400, 300) with the actions added back into the Q network after the first layer...

REPLAY_CAPACITY = 1_000_000
BATCH_SIZE = 256
START_TIMESTEPS = 25_000 # random warmup to fill the buffer

EXPL_NOISE = 0.1 # std of gaussian exploration noise
POLICY_NOISE = 0.2 #std of target-smoothing noise (in the target)
NOISE_CLIP = 0.5 #clip range for the smoothing noise
POLICY_DELAY = 2  #d, update actor + targets every d critic steps

MAX_STEPS = 1_000_000
EVAL_EVERY = 5_000
EVAL_EPISODES = 10
EVAL_MAX_STEPS = 1_000
NUM_EVAL_WORKERS = 5
LOG_EVERY = 1_000
EPOCH_SIZE = 50_000
Q_TRACK_STATES = 1_000 #fixed states for overestimation tracking

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


DEVICE = get_device()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN[0], HIDDEN[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN[1], action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        x = self.net(state)
        return x * self.max_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN[0], HIDDEN[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN[1], 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class TD3:
    def __init__(self, state_dim, action_dim, max_action, 
                 tau, gamma, 
                 policy_noise, noise_clip, policy_delay):
        
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.Q1 = Critic(state_dim, action_dim).to(DEVICE)
        self.Q2 = Critic(state_dim, action_dim).to(DEVICE)
        self.Q1_target = copy.deepcopy(self.Q1)
        self.Q2_target = copy.deepcopy(self.Q2)
        self.Q1_optimizer = torch.optim.Adam(self.Q1.parameters(), lr=LR_CRITIC)
        self.Q2_optimizer = torch.optim.Adam(self.Q2.parameters(), lr=LR_CRITIC)

        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.total_it = 0

    # acting during training to fill the replay buffer
    def select_action(self, obs, noise):
        with torch.no_grad():
            state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            action = self.actor(state).squeeze(0).cpu().numpy()
            action += np.random.normal(0, noise, size=action.shape)
            action = np.clip(action, -self.max_action, self.max_action)
        return action

    # the overestimation tracker (Fig. 1 analogue)
    # only tacking this as a sanity check to make sure values aren't exploding. 
    def estimate_value(self, q_track_states):
        with torch.no_grad():
            q_values = self.Q1(q_track_states, self.actor(q_track_states))
        return q_values.mean().item()

    def train(self, replay_buffer, batch_size):
        batch = replay_buffer.sample(batch_size) # state, action, reward, next_state, done
        critic_loss = self._update_critic(batch)

        actor_loss = None

        if self.total_it % self.policy_delay == 0:
            actor_loss = self._update_actor(batch)
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.Q1, self.Q1_target)
            self._soft_update(self.Q2, self.Q2_target)

        self.total_it += 1

        return {"critic_loss": critic_loss, "actor_loss": actor_loss}

    def _update_critic(self, batch):
        state, action, reward, next_state, done = batch
        reg_noise = torch.clip(torch.randn_like(action) * self.policy_noise, -self.noise_clip, self.noise_clip)
        with torch.no_grad():
            target_action = torch.clip(self.actor_target(next_state) + reg_noise, -self.max_action, self.max_action)
            target = reward + self.gamma * (1 - done) * torch.min(
                self.Q1_target(next_state, target_action),
                self.Q2_target(next_state, target_action)
            )
        # why not just add them up?
        critic_loss = F.mse_loss(self.Q1(state, action), target) + F.mse_loss(self.Q2(state, action), target)
        self.Q1_optimizer.zero_grad()
        self.Q2_optimizer.zero_grad()
        critic_loss.backward()
        self.Q1_optimizer.step()
        self.Q2_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, batch):
        state, action, reward, next_state, done = batch
        current_action = self.actor(state)
        actor_loss = -self.Q1(state, current_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _soft_update(self, online_module, target_module):
        for online_param, target_param in zip(online_module.parameters(), target_module.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1 - self.tau) * target_param.data)

    # checkpointing
    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "Q1": self.Q1.state_dict(), "Q2": self.Q2.state_dict(),
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "Q1_optimizer": self.Q1_optimizer.state_dict(),
                    "Q2_optimizer": self.Q2_optimizer.state_dict(),
                    "total_it": self.total_it}, path)

    def load(self, path):
        c = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(c["actor"])
        self.Q1.load_state_dict(c["Q1"]); self.Q2.load_state_dict(c["Q2"])
        self.actor_optimizer.load_state_dict(c["actor_optimizer"])
        self.Q1_optimizer.load_state_dict(c["Q1_optimizer"])
        self.Q2_optimizer.load_state_dict(c["Q2_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)
        self.Q1_target = copy.deepcopy(self.Q1); self.Q2_target = copy.deepcopy(self.Q2)
        self.total_it = c.get("total_it", 0)

#
# MuJoCo env wrapper
#
class MujocoEnv:
    """Thin Gymnasium wrapper. Surfaces terminated and truncated SEPARATELY so
    the bootstrap mask (terminated) and the reset decision (terminated OR
    truncated) can be handled correctly downstream."""

    def __init__(self, env_name=ENV_NAME, seed=SEED):
        self.env = gym.make(env_name)
        self._seed = seed
        self._seeded = False
        self.state_dim = int(self.env.observation_space.shape[0])
        self.action_dim = int(self.env.action_space.shape[0])
        # MuJoCo action bounds are symmetric, so high[0] gives max_action.
        self.max_action = float(self.env.action_space.high[0])

    def reset(self):
        if not self._seeded:
            obs, _ = self.env.reset(seed=self._seed)
            self._seeded = True
        else:
            obs, _ = self.env.reset()
        return obs.astype(np.float32)

    def step(self, action):
        obs, r, terminated, truncated, _ = self.env.step(action)
        return obs.astype(np.float32), float(r), bool(terminated), bool(truncated)

    def sample_action(self):
        return self.env.action_space.sample()

    def close(self):
        self.env.close()


# 
# Replay buffer
# 
class ReplayBuffer:

    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        t = lambda arr: torch.as_tensor(arr[idx], device=self.device)
        return (
            t(self.states),
            t(self.actions),
            t(self.rewards),
            t(self.next_states),
            t(self.dones),
        )


# 
# wandb setup
# 
def init_wandb():
    try:
        run = wandb.init(
            project="td3-mujoco",
            config={
                "env": ENV_NAME,
                "gamma": GAMMA,
                "tau": TAU,
                "lr_actor": LR_ACTOR,
                "lr_critic": LR_CRITIC,
                "hidden": HIDDEN,
                "replay_capacity": REPLAY_CAPACITY,
                "batch_size": BATCH_SIZE,
                "start_timesteps": START_TIMESTEPS,
                "expl_noise": EXPL_NOISE,
                "policy_noise": POLICY_NOISE,
                "noise_clip": NOISE_CLIP,
                "policy_delay": POLICY_DELAY,
                "max_steps": MAX_STEPS,
                "eval_every": EVAL_EVERY,
                "eval_episodes": EVAL_EPISODES,
                "seed": SEED,
                "device": str(DEVICE),
            },
        )
        # independent x-axis for eval so it never collides with the train step.
        wandb.define_metric("eval_step")
        wandb.define_metric("eval_return", step_metric="eval_step")
        return run
    except Exception as e:
        raise RuntimeError(
            f"W&B initialisation failed: {e}\n"
            "Run `wandb login` and ensure the `wandb` package is installed."
        ) from e


# 
# Overestimation tracking: fixed random-policy states
# an LLM suggested that i track this just because
# 
def collect_q_track_states(env_name=ENV_NAME, n=Q_TRACK_STATES, seed=SEED):
    """Roll out a random policy to gather a fixed bank of states for the
    overestimation tracker (cf. Fig. 1 in the TD3 paper)."""
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed + 1)
    states = []
    while len(states) < n:
        a = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(a)
        states.append(obs.astype(np.float32))
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return torch.as_tensor(np.array(states[:n]), dtype=torch.float32)


# 
# Evaluation 
# deterministic policy, no exploration noise
# 
def run_single_eval_episode(actor_cpu, env_name, max_action, seed):
    """One eval episode using the deterministic actor on CPU. Calls the actor
    directly: action = actor(state). No exploration noise."""
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    ep_ret = 0.0
    steps = 0
    while steps < EVAL_MAX_STEPS:
        state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = actor_cpu(state).squeeze(0).numpy()
        obs, r, terminated, truncated, _ = env.step(action)
        ep_ret += r
        steps += 1
        if terminated or truncated:
            break
    env.close()
    return ep_ret


def evaluate(actor_cpu, env_name, max_action, episodes=EVAL_EPISODES):
    """Run eval episodes in parallel; return the mean return."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    seeds = [SEED + 10_000 + i for i in range(episodes)]
    returns = []
    with ThreadPoolExecutor(max_workers=NUM_EVAL_WORKERS) as ex:
        futures = [
            ex.submit(run_single_eval_episode, actor_cpu, env_name, max_action, s)
            for s in seeds
        ]
        for f in futures:
            returns.append(f.result())
    torch.set_num_threads(old_threads)
    return float(np.mean(returns))


class EvalWorker:
    """Runs evaluate() in a background thread on a CPU snapshot of the actor, so
    the training device is never blocked or contended."""

    def __init__(self, agent, env_name, max_action):
        self._agent = agent
        self._env_name = env_name
        self._max_action = max_action
        self._thread = None
        self._result = None
        self._step = None

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, step):
        if self.is_running():
            return
        actor_cpu = copy.deepcopy(self._agent.actor).to("cpu").eval()
        self._step = step
        self._result = None

        def _run():
            self._result = evaluate(actor_cpu, self._env_name, self._max_action)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def try_collect(self):
        if self._thread is not None and not self._thread.is_alive():
            result, step = self._result, self._step
            self._thread = self._result = self._step = None
            return step, result
        return None


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    run = init_wandb()

    env = MujocoEnv(ENV_NAME, SEED)
    state_dim, action_dim, max_action = env.state_dim, env.action_dim, env.max_action
    tqdm.write(
        f"env={ENV_NAME}  state_dim={state_dim}  action_dim={action_dim}  "
        f"max_action={max_action}  device={DEVICE}"
    )

    agent = TD3(state_dim, action_dim, max_action, TAU, GAMMA, POLICY_NOISE, NOISE_CLIP, POLICY_DELAY)
    buffer = ReplayBuffer(REPLAY_CAPACITY, state_dim, action_dim, DEVICE)

    tqdm.write("Collecting fixed states for overestimation tracking...")
    q_track_states = collect_q_track_states().to(DEVICE)
    tqdm.write(f"Collected {len(q_track_states)} states.")

    obs = env.reset()
    ep_ret, ep_len = 0.0, 0
    recent_returns = deque(maxlen=20)
    critic_losses = deque(maxlen=100)
    actor_losses = deque(maxlen=100)
    best_eval = -float("inf")
    eval_worker = EvalWorker(agent, ENV_NAME, max_action)

    pbar = tqdm(range(1, MAX_STEPS + 1), dynamic_ncols=True)
    t0 = time.time()

    # Immediate eval to confirm the pipeline end-to-end.
    eval_worker.start(step=0)
    tqdm.write("[step 0] Launched immediate background evaluation...")

    for step in pbar:
        epoch = (step - 1) // EPOCH_SIZE + 1

        # action selection: random warmup, then policy + noise
        if step <= START_TIMESTEPS:
            action = env.sample_action()
        else:
            action = agent.select_action(obs, EXPL_NOISE)

        next_obs, r, terminated, truncated, = env.step(action)
        # for half cheetah there is no termination. the value of the next state still needs to be computed.
        # therefore, truncated is not really "done". can ignore it for the memory buffer. 
        buffer.push(obs, action, r, next_obs, float(terminated))
        obs = next_obs
        ep_ret += r
        ep_len += 1

        if terminated or truncated:
            recent_returns.append(ep_ret)
            obs = env.reset()
            ep_ret, ep_len = 0.0, 0

        # training: one iteration per env step after warmup
        if step > START_TIMESTEPS:
            stats = agent.train(buffer, BATCH_SIZE)
            if stats is not None:
                if stats.get("critic_loss") is not None:
                    critic_losses.append(float(stats["critic_loss"]))
                if stats.get("actor_loss") is not None:
                    actor_losses.append(float(stats["actor_loss"]))

        # --- Logging ---
        if step % LOG_EVERY == 0:
            avg_ret = float(np.mean(recent_returns)) if recent_returns else 0.0
            avg_closs = float(np.mean(critic_losses)) if critic_losses else 0.0
            avg_aloss = float(np.mean(actor_losses)) if actor_losses else 0.0
            sps = step / (time.time() - t0)
            avg_q = agent.estimate_value(q_track_states)
            pbar.set_postfix({
                "epoch": epoch,
                "ret": f"{avg_ret:.1f}",
                "c_loss": f"{avg_closs:.3f}",
                "a_loss": f"{avg_aloss:.3f}",
                "Q": f"{avg_q:.2f}",
                "buf": buffer.size,
                "sps": f"{sps:.0f}",
            })
            wandb.log(
                {
                    "epoch": epoch,
                    "avg_return": avg_ret,
                    "critic_loss": avg_closs,
                    "actor_loss": avg_aloss,
                    "avg_q": avg_q,
                    "buffer_size": buffer.size,
                    "sps": sps,
                },
                step=step,
            )

        # --- Collect a finished background eval ---
        result = eval_worker.try_collect()
        if result is not None:
            eval_step, eval_ret = result
            tqdm.write(f"[step {eval_step}] eval return: {eval_ret:.1f}")
            wandb.log({
                "eval_return": eval_ret,
                "eval_step": eval_step if eval_step > 0 else step,
            })
            if eval_ret > best_eval:
                best_eval = eval_ret
                ckpt = os.path.join(os.path.dirname(run.dir), "td3_best.pt")
                agent.save(ckpt)
                tqdm.write(f"Saved best checkpoint: {ckpt}")

        # --- Launch a new background eval ---
        if step % EVAL_EVERY == 0 and step > START_TIMESTEPS:
            eval_worker.start(step=step)

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()