import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"  # avoid video recording error in headless server

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import gymnasium

import torch
from torch import nn
from torch import distributions as torchd
from tqdm.auto import tqdm

# Metaworld import setup
import envs.metaworld_wrappers as metaworld_wrappers
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import metaworld
from metaworld.wrappers import ProprioImageObsWrapper, ProprioMultiImageObsWrapper


to_np = lambda x: x.detach().cpu().numpy()


# ===========================================================================
#  Dreamer agent – thin orchestrator over disentangled components
# ===========================================================================
# The agent holds three independent nn.Modules:
#   1. _rssm         – shared RSSM backbone  (RSSMWorldModel)
#   2. _task_heads    – per-task reward & cont heads  (TaskHeads)
#   3. _actor_critic  – per-task actor & value nets  (ActorCritic)
#
# Each can be saved / loaded independently via tools.save_component /
# tools.load_component, enabling the continual-learning workflow:
#   train task 1 → save heads+ac → init fresh heads+ac → train task 2 → ...
# ---------------------------------------------------------------------------


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset

        # --- 1. Shared RSSM backbone (encoder, dynamics, decoder) ---
        self._rssm = models.RSSMWorldModel(obs_space, act_space, self._step, config)

        # --- 2. Task-specific heads (reward, continuation) ---
        self._task_heads = models.TaskHeads(config)

        # --- 3. Task-specific actor-critic ---
        self._actor_critic = models.ActorCritic(config)

        # Optional torch.compile
        if config.compile and os.name != "nt":
            self._rssm = torch.compile(self._rssm)
            self._task_heads = torch.compile(self._task_heads)
            self._actor_critic = torch.compile(self._actor_critic)

        # --- Exploration behavior ---
        # The reward function for exploration uses the task heads
        reward = lambda f, s, a: self._task_heads.reward(f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._actor_critic,
            random=lambda: expl.Random(config, act_space),
            # Plan2Explore still uses the old ImagBehavior interface.
            # For single-task training with greedy exploration this is unused.
            plan2explore=lambda: expl.Plan2Explore(config, self._rssm, reward),
        )[config.expl_behavior]()
        if isinstance(self._expl_behavior, nn.Module):
            self._expl_behavior = self._expl_behavior.to(self._config.device)

    # ---- called by tools.simulate during rollout --------------------------

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._rssm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    # ---- policy: latent inference + action selection ----------------------

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._rssm.preprocess(obs)
        embed = self._rssm.encoder(obs)
        latent, _ = self._rssm.dynamics.obs_step(
            latent, action, embed, obs["is_first"]
        )
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._rssm.dynamics.get_feat(latent)

        if not training:
            actor = self._actor_critic.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._actor_critic.actor(feat)
            action = actor.sample()

        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    # ---- one training step ------------------------------------------------

    def _train(self, data):
        metrics = {}

        # 1. Train shared RSSM + task heads jointly
        post, context, mets = models.train_world_model_step(
            self._rssm, self._task_heads, data
        )
        metrics.update(mets)

        # 2. Train actor-critic on imagined trajectories
        #    The objective (reward) comes from the task-specific reward head
        #    applied to the latent features from the RSSM.
        start = post

        def reward_fn(feat, state, action):
            return self._task_heads.reward(
                self._rssm.get_feat(state)
            ).mode()

        ac_results = self._actor_critic._train(
            start, reward_fn, self._rssm, self._task_heads
        )
        metrics.update(ac_results[-1])

        # 3. Exploration behavior training (no-op for greedy)
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})

        for name, value in metrics.items():
            if name not in self._metrics:
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


# ===========================================================================
#  Helper functions (unchanged from original)
# ===========================================================================

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "metaworld":
        env = gymnasium.make(
            "Meta-World/MT1",
            env_name=task,
            render_mode="rgb_array",
            max_episode_steps=config.time_limit,
        )
        env = ProprioMultiImageObsWrapper(
            env,
            image_height=config.size[0],
            image_width=config.size[1],
            camera_names=["topview", "front", "gripperPOV"],
        )
        env = metaworld_wrappers.FirstTerminalObs(env)
        env = metaworld_wrappers.RewardTuningWrapperV2(env)
        env = metaworld_wrappers.Gymnasium2Gym(env)
        env = wrappers.NormalizeActions(env)
        env = wrappers.RewardObs(env)
        env = wrappers.TimeLimit(env, config.time_limit)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env)
        return env

    print("You are running original DreamerV3 code, not using Metaworld.")

    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


class LazyParallelEnv:
    """Picklable proxy that lazily constructs the real env inside the worker."""

    def __init__(self, config, mode, env_id):
        self._config = config
        self._mode = mode
        self._env_id = env_id
        self._env = None

    def _ensure_env(self):
        if self._env is None:
            self._env = make_env(self._config, self._mode, self._env_id)
        return self._env

    @property
    def observation_space(self):
        return self._ensure_env().observation_space

    @property
    def action_space(self):
        return self._ensure_env().action_space

    @property
    def id(self):
        return self._ensure_env().id

    def reset(self):
        return self._ensure_env().reset()

    def step(self, action):
        return self._ensure_env().step(action)

    def close(self):
        if self._env is not None:
            return self._env.close()


# ===========================================================================
#  Checkpoint helpers – save / load the three components independently
# ===========================================================================

def save_checkpoint(agent, logdir, step):
    """Save all three components as separate files + a manifest."""
    ckpt_dir = pathlib.Path(logdir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tools.save_component(agent._rssm, ckpt_dir / "rssm.pt", step=step)
    tools.save_component(agent._task_heads, ckpt_dir / "task_heads.pt", step=step)
    tools.save_component(agent._actor_critic, ckpt_dir / "actor_critic.pt", step=step)
    # Also save a small manifest with the training step and metadata
    # so we can quickly check state without loading full tensors.
    manifest = {"step": step}
    torch.save(manifest, ckpt_dir / "manifest.pt")
    print(f"Checkpoint saved at step {step}")


def load_checkpoint(agent, logdir, load_optimizers=True, device=None):
    """Load all three components from a checkpoint directory.

    Args:
        agent:           Dreamer agent instance (with matching architecture).
        logdir:          path containing checkpoints/ subdirectory.
        load_optimizers: if True, restore optimizer states for resumed training.
        device:          optional device mapping.

    Returns:
        step from the checkpoint manifest.
    """
    ckpt_dir = pathlib.Path(logdir) / "checkpoints"
    tools.load_component(
        agent._rssm, ckpt_dir / "rssm.pt",
        load_optimizers=load_optimizers, device=device,
    )
    tools.load_component(
        agent._task_heads, ckpt_dir / "task_heads.pt",
        load_optimizers=load_optimizers, device=device,
    )
    tools.load_component(
        agent._actor_critic, ckpt_dir / "actor_critic.pt",
        load_optimizers=load_optimizers, device=device,
    )
    manifest = torch.load(ckpt_dir / "manifest.pt", map_location=device)
    print(f"Checkpoint loaded from {ckpt_dir}  (step={manifest['step']})")
    return manifest["step"]


# ===========================================================================
#  Main training loop
# ===========================================================================

def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)

    # --- Logger ---
    if args.logger == "tensorboard":
        logger = tools.Logger(logdir, config.action_repeat * step)
    elif args.logger == "wandb":
        logger = tools.WandBLogger(args, config, logdir, config.action_repeat * step)
    else:
        raise NotImplementedError(f"Logger {args.logger} is not implemented.")

    # --- Print training info ---
    print(">>> DREAMERV3E: Task Setup Configuration: <<<")
    print(f"Task: {config.task}")
    print(f"Image observation size: {config.size}")
    print(f"Action repeat: {config.action_repeat}")
    print(f"Time limit (in env step): {config.time_limit}")
    print("================================")
    print(">>> DREAMERV3: Training Configuration: <<<")
    print(f"Total training steps (in env step): {config.steps * config.action_repeat}")
    print(f"Evaluation every (in env step): {config.eval_every * config.action_repeat}")
    print(f"Logging every (in env step): {config.log_every * config.action_repeat}")
    print(f"Number of parallel environments: {config.envs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Train ratio: {config.train_ratio}")
    print(f"Video prediction logging: {config.video_pred_log}")
    print(f"Pretraining steps (in env step): {config.pretrain * config.action_repeat}")
    print(f"Exploration until (in env step): {config.expl_until * config.action_repeat}")
    print(f"Exploration behavior: {config.expl_behavior}")
    print(f"Evaluation episodes: {config.eval_episode_num}")
    print("================================")
    if config.skip_config_check:
        print(">>> DREAMERV3: Start training...")
    else:
        input(">>> DREAMERV3: Press Enter to start training...")

    # --- Load episode data ---
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)

    # --- Create environments ---
    if config.parallel:
        train_envs = [
            Parallel(LazyParallelEnv(config, "train", i), "process")
            for i in range(config.envs)
        ]
    else:
        make = lambda mode, id: make_env(config, mode, id)
        train_envs = [make("train", i) for i in range(config.envs)]
        train_envs = [Damy(env) for env in train_envs]
    acts = train_envs[0].action_space
    print(">>> DREAMERV3: Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # --- Prefill replay buffer with random actions ---
    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f">>> DREAMERV3: Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f">>> DREAMERV3: Logger: ({logger.step} steps).")

    # --- Build agent ---
    print(">>> DREAMERV3: Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    # --- Load checkpoint if available ---
    ckpt_dir = logdir / "checkpoints"
    if config.from_checkpoint is not None:
        # Load from explicitly specified checkpoint
        if os.path.isdir(config.from_checkpoint):
            ckpt_path = pathlib.Path(config.from_checkpoint)
        else:
            raise FileNotFoundError(
                f"Checkpoint directory {config.from_checkpoint} does not exist."
            )
        print(f">>> DREAMERV3: Loading from specified checkpoint: {ckpt_path}")
        load_checkpoint(agent, ckpt_path.parent, load_optimizers=True)
        if config.skip_pretrain:
            agent._should_pretrain._once = False
    elif (ckpt_dir / "manifest.pt").exists():
        # Resume from current logdir
        print(f">>> DREAMERV3: Resuming from {ckpt_dir}")
        load_checkpoint(agent, logdir, load_optimizers=True)
        agent._should_pretrain._once = False

    # --- Training loop ---
    train_steps_total = config.steps
    train_steps_done = min(agent._step, train_steps_total)
    progress_bar = tqdm(
        total=train_steps_total,
        initial=train_steps_done,
        desc=">>> DREAMERV3: Training Progress",
        unit="step",
    )
    try:
        while agent._step < config.steps + config.eval_every:
            logger.write()

            # --- Evaluation ---
            if config.eval_episode_num > 0:
                print(">>> DREAMERV3: Start evaluation.")
                if config.parallel:
                    eval_envs = [
                        Parallel(LazyParallelEnv(config, "eval", i), "process")
                        for i in range(config.envs)
                    ]
                else:
                    eval_envs = [
                        Damy(make_env(config, "eval", i))
                        for i in range(config.envs)
                    ]
                eval_policy = functools.partial(agent, training=False)
                tools.simulate(
                    eval_policy,
                    eval_envs,
                    eval_eps,
                    config.evaldir,
                    logger,
                    is_eval=True,
                    episodes=config.eval_episode_num,
                )
                if config.video_pred_log:
                    video_pred = agent._rssm.video_pred(next(eval_dataset))
                    logger.video("eval_openl", to_np(video_pred))
                for env in eval_envs:
                    try:
                        env.close()
                    except Exception:
                        pass
                del eval_envs

            # --- Training ---
            print(">>> DREAMERV3: Start training.")
            state = tools.simulate(
                agent,
                train_envs,
                train_eps,
                config.traindir,
                logger,
                limit=config.dataset_size,
                steps=config.eval_every,
                state=state,
            )

            # Update progress bar
            updated_train_steps = min(agent._step, train_steps_total)
            step_delta = max(0, updated_train_steps - train_steps_done)
            if step_delta:
                progress_bar.update(step_delta)
                train_steps_done = updated_train_steps

            # Save checkpoint – each component is a separate file
            save_checkpoint(agent, logdir, step=agent._step)

    finally:
        progress_bar.close()

    for env in train_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    # Specify logger
    parser.add_argument(
        "--logger", type=str, default="wandb"
    )  # options: wandb, tensorboard
    # Wandb arguments
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="CCLB_Dreamerv3")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    # Extra arguments for checkpoint control
    parser.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to load weights from.",
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip pretraining when loading from checkpoint.",
    )
    parser.add_argument(
        "--skip-config-check",
        action="store_true",
        help="Skip the interactive config confirmation prompt.",
    )
    main(parser.parse_args(remaining))
