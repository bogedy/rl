import os
import random
import threading
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from ale_py import ALEInterface, roms
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import wandb

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
DTYPE = torch.bfloat16
DEVICE = torch.device("xpu")
SEED = 42
FRAME_STACK = 4
RESIZE_H, RESIZE_W = 84, 84          # 84×84 (Nature DQN paper)
REPLAY_CAPACITY = 1_000_000
MIN_REPLAY_SIZE = 50_000
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY_STEPS = 1_000_000
TARGET_UPDATE_FREQ = 10_000
TRAIN_FREQ = 4
MAX_STEPS = 10_000_000
EVAL_EVERY = 10_000
EVAL_EPS = 0.05
EVAL_EPISODES = 10                    # 10 eval episodes as per notes
NUM_EVAL_WORKERS = 6                  # Parallel evaluation workers (matches P-cores)
NOOP_MAX = 30
FRAME_SKIP = 4
GRAD_CLIP = 10.0
LOG_EVERY = 1_000
EPOCH_SIZE = 50_000                   # steps per epoch (paper convention)
Q_TRACK_STATES = 1_000               # fixed random-policy states for avg-max-Q

# Crop coordinates (Beam Rider gameplay area)
ROW_START, ROW_END = 30, 190
COL_START, COL_END = 8, 160

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ---------------------------------------------------------------------------
# W&B startup check
# ---------------------------------------------------------------------------
def init_wandb():
    """Initialise W&B; raise clearly if it is not configured."""
    try:
        run = wandb.init(
            project="dqn-atari",
            config={
                "game": "beam_rider",
                "frame_stack": FRAME_STACK,
                "resize": (RESIZE_H, RESIZE_W),
                "crop": (ROW_START, ROW_END, COL_START, COL_END),
                "replay_capacity": REPLAY_CAPACITY,
                "batch_size": BATCH_SIZE,
                "gamma": GAMMA,
                "lr": LR,
                "eps_start": EPS_START,
                "eps_end": EPS_END,
                "eps_decay_steps": EPS_DECAY_STEPS,
                "target_update_freq": TARGET_UPDATE_FREQ,
                "train_freq": TRAIN_FREQ,
                "max_steps": MAX_STEPS,
                "eval_every": EVAL_EVERY,
                "eval_episodes": EVAL_EPISODES,
                "epoch_size": EPOCH_SIZE,
                "seed": SEED,
            },
        )
        # Give eval_return its own independent x-axis so it never conflicts
        # with the training step counter (avoids W&B monotonic-step rejections).
        wandb.define_metric("eval_step")
        wandb.define_metric("eval_return", step_metric="eval_step")
        return run
    except Exception as e:
        raise RuntimeError(
            f"W&B initialisation failed: {e}\n"
            "Make sure you have run `wandb login` and that the `wandb` package is installed."
        ) from e


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
class Preprocessor:
    """Crop → grayscale → resize to 56×56 → uint8 CPU tensor."""

    def __call__(self, frame_np: np.ndarray) -> torch.Tensor:
        # frame_np: (210, 160, 3) uint8
        cropped = frame_np[ROW_START:ROW_END, COL_START:COL_END]  # (150, 110, 3)
        # Convert to PIL, grayscale, resize
        pil = TF.to_pil_image(cropped)
        pil = TF.to_grayscale(pil)
        pil = TF.resize(pil, (RESIZE_H, RESIZE_W))
        t = TF.pil_to_tensor(pil)  # (1, 56, 56) uint8
        return t.squeeze(0)         # (56, 56) uint8 CPU


# ---------------------------------------------------------------------------
# Atari environment
# ---------------------------------------------------------------------------
class AtariEnv:
    def __init__(self, rom_name="beam_rider", frame_skip=FRAME_SKIP, noop_max=NOOP_MAX, seed=SEED):
        self.ale = ALEInterface()
        self.ale.setInt("random_seed", seed)
        self.ale.setFloat("repeat_action_probability", 0.0)
        self.ale.setBool("color_averaging", False)
        self.ale.loadROM(roms.get_rom_path(rom_name))
        self.actions = self.ale.getMinimalActionSet()
        self.frame_skip = frame_skip
        self.noop_max = noop_max
        self._obs_buf = np.zeros((2, 210, 160, 3), dtype=np.uint8)

    @property
    def num_actions(self):
        return len(self.actions)

    def reset(self):
        self.ale.reset_game()
        noops = random.randint(1, self.noop_max)
        for _ in range(noops):
            self.ale.act(0)
            if self.ale.game_over():
                self.ale.reset_game()
        self.ale.getScreenRGB(self._obs_buf[0])
        self._obs_buf[1] = self._obs_buf[0]
        return self._obs_buf[0]

    def step(self, action_idx):
        action = self.actions[action_idx]
        total_reward = 0.0
        done = False
        for i in range(self.frame_skip):
            r = self.ale.act(action)
            total_reward += r
            if i == self.frame_skip - 2:
                self.ale.getScreenRGB(self._obs_buf[0])
            elif i == self.frame_skip - 1:
                self.ale.getScreenRGB(self._obs_buf[1])
            if self.ale.game_over():
                done = True
                break
        frame = np.maximum(self._obs_buf[0], self._obs_buf[1])
        return frame, total_reward, done


# ---------------------------------------------------------------------------
# Replay buffer  –  stores uint8 frames, casts to bf16 on the fly
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        # Compact storage: uint8 on CPU
        self.frames = torch.zeros((capacity, RESIZE_H, RESIZE_W), dtype=torch.uint8)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.bool)
        self.pos = 0
        self.size = 0

    def push(self, frame_u8_cpu, action, reward, done):
        self.frames[self.pos] = frame_u8_cpu
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # Oversample and filter out candidates that:
        # 1. Cross an episode boundary (any dones[idx-4] to dones[idx-1] is True)
        # 2. Straddle the circular buffer's write head (self.pos) when the buffer is full
        valid = torch.zeros(0, dtype=torch.long)
        lookback_offs = torch.arange(-FRAME_STACK, 0)      # [-4, -3, -2, -1]
        pos_offs = torch.arange(-FRAME_STACK, 1)           # [-4, -3, -2, -1, 0]

        while valid.shape[0] < batch_size:
            if self.size < self.capacity:
                # Buffer is not full: sample from [FRAME_STACK, self.size - 1].
                # No write-head straddling is possible since self.pos == self.size.
                candidates = torch.randint(FRAME_STACK, self.size, (batch_size * 4,))
                lb_idx = (candidates.unsqueeze(1) + lookback_offs.unsqueeze(0)) % self.capacity
                crosses = self.dones[lb_idx].any(dim=1)
                valid = torch.cat([valid, candidates[~crosses]])
            else:
                # Buffer is full: sample from [0, self.capacity - 1].
                candidates = torch.randint(0, self.capacity, (batch_size * 4,))
                
                # Filter 1: Check if any accessed index of the 5-frame transition span matches self.pos
                pos_idx = (candidates.unsqueeze(1) + pos_offs.unsqueeze(0)) % self.capacity
                straddles = (pos_idx == self.pos).any(dim=1)
                
                # Filter 2: Check if any lookback frame was terminal
                lb_idx = (candidates.unsqueeze(1) + lookback_offs.unsqueeze(0)) % self.capacity
                crosses = self.dones[lb_idx].any(dim=1)
                
                is_valid = ~straddles & ~crosses
                valid = torch.cat([valid, candidates[is_valid]])

        idx = valid[:batch_size]
        # State offsets: [-4, -3, -2, -1]
        s_offs = torch.arange(-FRAME_STACK, 0)
        s_idx = (idx.unsqueeze(1) + s_offs.unsqueeze(0)) % self.capacity
        # Next state offsets: [-3, -2, -1, 0]
        ns_idx = (s_idx + 1) % self.capacity
        
        states = (
            self.frames[s_idx]
            .to(self.device, non_blocking=True)
            .to(DTYPE) / 255.0
        )
        next_states = (
            self.frames[ns_idx]
            .to(self.device, non_blocking=True)
            .to(DTYPE) / 255.0
        )
        actions = self.actions[idx].to(self.device, non_blocking=True)
        rewards = self.rewards[idx].to(self.device, non_blocking=True)
        dones = self.dones[idx].to(self.device, non_blocking=True)
        return states, actions, rewards, next_states, dones



# ---------------------------------------------------------------------------
# Frame stacker
# ---------------------------------------------------------------------------
class FrameStacker:
    def __init__(self):
        self.stack = deque(maxlen=FRAME_STACK)

    def reset(self, frame_u8_cpu):
        self.stack.clear()
        for _ in range(FRAME_STACK):
            self.stack.append(frame_u8_cpu)

    def push(self, frame_u8_cpu):
        self.stack.append(frame_u8_cpu)

    def get_state(self, device):
        # (4, 56, 56) uint8 → (1, 4, 56, 56) bf16 on device
        s = torch.stack(list(self.stack), dim=0).unsqueeze(0)
        return s.to(device, non_blocking=True).to(DTYPE) / 255.0


# ---------------------------------------------------------------------------
# Q-network  –  adjusted FC size for 56×56 input
# ---------------------------------------------------------------------------
class QNet(nn.Module):
    """
    Conv output for 84×84 input:
      Conv1: (84-8)/4+1 = 20  → 20×20
      Conv2: (20-4)/2+1 =  9  →  9×9
      Conv3: (9-3)/1+1  =  7  →  7×7
      Flat: 64 * 7 * 7 = 3136
    """

    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(FRAME_STACK, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def epsilon_at(step):
    frac = min(1.0, step / EPS_DECAY_STEPS)
    return EPS_START + frac * (EPS_END - EPS_START)


def collect_q_track_states(preprocessor, n=Q_TRACK_STATES):
    """Run a random policy to collect a fixed set of states for Q tracking."""
    env = AtariEnv(seed=SEED + 1)
    stacker = FrameStacker()
    frame = env.reset()
    stacker.reset(preprocessor(frame))
    states = []
    while len(states) < n:
        a = random.randrange(env.num_actions)
        frame, _, done = env.step(a)
        f = preprocessor(frame)
        stacker.push(f)
        states.append(torch.stack(list(stacker.stack), dim=0))  # (4,56,56) uint8
        if done:
            frame = env.reset()
            stacker.reset(preprocessor(frame))
    # (N, 4, 56, 56) uint8 – kept on CPU to save device memory
    return torch.stack(states[:n], dim=0)


@torch.no_grad()
def avg_max_q(qnet, q_states_cpu, device, chunk=64):
    """Average of max Q-value over the fixed state set."""
    qnet.eval()
    total = 0.0
    for i in range(0, len(q_states_cpu), chunk):
        batch = q_states_cpu[i : i + chunk].to(device, non_blocking=True).to(DTYPE) / 255.0
        with torch.autocast(device_type="xpu", dtype=DTYPE):
            q = qnet(batch)
        total += q.max(1).values.sum().item()
    qnet.train()
    return total / len(q_states_cpu)


def run_single_eval_episode(qnet, device, preprocessor, eps, seed):
    """Run a single evaluation episode in a thread-safe manner."""
    env = AtariEnv(seed=seed)
    stacker = FrameStacker()
    frame = env.reset()
    stacker.reset(preprocessor(frame))
    ep_ret = 0.0
    done = False
    steps = 0
    cpu_dtype = torch.float32
    while not done and steps < 27000:
        if random.random() < eps:
            a = random.randrange(env.num_actions)
        else:
            # Build state on the eval device (CPU) in float32
            s = torch.stack(list(stacker.stack), dim=0).unsqueeze(0)
            state = s.to(device).to(cpu_dtype) / 255.0
            q = qnet(state)
            a = int(q.argmax(1).item())
        frame, r, done = env.step(a)
        stacker.push(preprocessor(frame))
        ep_ret += r
        steps += 1
    return ep_ret


@torch.no_grad()
def evaluate(qnet, device, preprocessor, eps=EVAL_EPS, episodes=EVAL_EPISODES):
    """Run evaluation episodes in parallel and return mean total reward."""
    # Force PyTorch to use exactly 1 CPU thread per evaluation run
    # to avoid OpenMP thread over-subscription and CPU thrashing.
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    total_returns = []
    # Seed each thread uniquely to guarantee different exploration/starting paths
    seeds = [SEED + 10000 + i for i in range(episodes)]

    with ThreadPoolExecutor(max_workers=NUM_EVAL_WORKERS) as executor:
        futures = [
            executor.submit(run_single_eval_episode, qnet, device, preprocessor, eps, seeds[i])
            for i in range(episodes)
        ]
        for f in futures:
            total_returns.append(f.result())

    # Restore original thread setting for any CPU operations in the main thread
    torch.set_num_threads(old_threads)
    return float(np.mean(total_returns))


# ---------------------------------------------------------------------------
# Background evaluation helper
# ---------------------------------------------------------------------------
class EvalWorker:
    """Runs evaluate() in a background thread so training is not blocked.

    The eval net intentionally stays on CPU so it never contends with the
    XPU training loop for device resources.
    """

    def __init__(self, qnet, preprocessor):
        self._qnet = qnet
        self._prep = preprocessor
        self._thread: threading.Thread | None = None
        self._result: float | None = None
        self._step: int | None = None

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, step: int):
        """Snapshot the current weights and launch evaluation in the background."""
        if self.is_running():
            return  # Don't start a second evaluation if one is already running
        # Deep-copy weights to CPU so the thread has a stable, XPU-free snapshot
        state_dict = {k: v.cpu().clone() for k, v in self._qnet.state_dict().items()}
        self._step = step
        self._result = None

        def _run():
            # Net stays on CPU — no XPU context needed, no device contention
            net = QNet(self._qnet.fc2.out_features)
            net.load_state_dict(state_dict)
            net.eval()
            self._result = evaluate(net, torch.device("cpu"), self._prep)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def try_collect(self):
        """Return (step, result) if the background eval finished, else None."""
        if self._thread is not None and not self._thread.is_alive():
            result = self._result
            step = self._step
            self._thread = None
            self._result = None
            self._step = None
            return step, result
        return None


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    # --- W&B check (fail fast) ---
    run = init_wandb()

    device = DEVICE
    env = AtariEnv()
    num_actions = env.num_actions

    preprocessor = Preprocessor()
    qnet = QNet(num_actions).to(device)
    target = QNet(num_actions).to(device)
    target.load_state_dict(qnet.state_dict())
    for p in target.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(qnet.parameters(), lr=LR)
    buffer = ReplayBuffer(REPLAY_CAPACITY, device)
    stacker = FrameStacker()

    # Collect fixed states for avg-max-Q tracking (random policy)
    tqdm.write("Collecting fixed states for Q tracking...")
    q_track_states = collect_q_track_states(preprocessor)
    tqdm.write(f"Collected {len(q_track_states)} Q-tracking states.")

    frame = env.reset()
    f56 = preprocessor(frame)
    stacker.reset(f56)
    buffer.push(f56, 0, 0.0, False)

    ep_ret = 0.0
    ep_len = 0
    recent_returns: deque = deque(maxlen=20)
    losses: deque = deque(maxlen=100)
    best_eval = -float("inf")
    eval_worker = EvalWorker(qnet, preprocessor)

    pbar = tqdm(range(1, MAX_STEPS + 1), dynamic_ncols=True)
    t0 = time.time()

    # Kick off an immediate evaluation so we can confirm everything works
    eval_worker.start(step=0)
    tqdm.write("[step 0] Launched immediate background evaluation...")

    for step in pbar:
        epoch = (step - 1) // EPOCH_SIZE + 1
        eps = epsilon_at(step)

        # --- Action selection ---
        if random.random() < eps or buffer.size < MIN_REPLAY_SIZE:
            a = random.randrange(num_actions)
        else:
            state = stacker.get_state(device)
            with torch.no_grad(), torch.autocast(device_type="xpu", dtype=DTYPE):
                q = qnet(state)
            a = int(q.argmax(1).item())

        frame, r, done = env.step(a)
        f56 = preprocessor(frame)
        stacker.push(f56)
        # Use raw unclipped score instead of reward clipping (r instead of clipped_r)
        buffer.push(f56, a, float(r), done)
        ep_ret += r
        ep_len += 1

        if done:
            recent_returns.append(ep_ret)
            frame = env.reset()
            f56 = preprocessor(frame)
            stacker.reset(f56)
            ep_ret = 0.0
            ep_len = 0

        # --- Training step ---
        if buffer.size >= MIN_REPLAY_SIZE and step % TRAIN_FREQ == 0:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
            with torch.autocast(device_type="xpu", dtype=DTYPE):
                q_vals = qnet(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target(next_states).max(1).values
                    tgt = rewards + GAMMA * next_q * (~dones).to(DTYPE)
                loss = F.mse_loss(q_vals, tgt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), GRAD_CLIP)
            optimizer.step()
            losses.append(loss.item())

        # --- Target network update ---
        if step % TARGET_UPDATE_FREQ == 0:
            target.load_state_dict(qnet.state_dict())

        # --- Logging ---
        if step % LOG_EVERY == 0:
            avg_ret = float(np.mean(recent_returns)) if recent_returns else 0.0
            avg_loss = float(np.mean(losses)) if losses else 0.0
            sps = step / (time.time() - t0)
            avg_q = avg_max_q(qnet, q_track_states, device)
            pbar.set_postfix({
                "epoch": epoch,
                "eps": f"{eps:.3f}",
                "ret": f"{avg_ret:.1f}",
                "loss": f"{avg_loss:.4f}",
                "buf": buffer.size,
                "sps": f"{sps:.0f}",
                "Q": f"{avg_q:.3f}",
            })
            wandb.log(
                {
                    "epoch": epoch,
                    "eps": eps,
                    "avg_return": avg_ret,
                    "loss": avg_loss,
                    "sps": sps,
                    "avg_max_q": avg_q,
                    "buffer_size": buffer.size,
                },
                step=step,
            )

        # --- Collect finished background evaluation ---
        result = eval_worker.try_collect()
        if result is not None:
            eval_step, eval_ret = result
            tqdm.write(f"[step {eval_step}] eval return: {eval_ret:.1f} (eps={EVAL_EPS})")
            # Log with eval_step as the x-axis (defined via define_metric in
            # init_wandb). This is completely independent of the training step
            # counter so W&B never rejects it for being out-of-order.
            wandb.log({"eval_return": eval_ret, "eval_step": eval_step if eval_step > 0 else step})
            if eval_ret > best_eval:
                best_eval = eval_ret
                ckpt_path = os.path.join(os.path.dirname(run.dir), "dqn_beamrider_best.pt")
                torch.save(qnet.state_dict(), ckpt_path)
                tqdm.write(f"Saved best model checkpoint to local run dir: {ckpt_path}")

        # --- Launch new background evaluation every EVAL_EVERY steps ---
        if step % EVAL_EVERY == 0 and buffer.size >= MIN_REPLAY_SIZE:
            eval_worker.start(step=step)

    wandb.finish()


if __name__ == "__main__":
    main()
