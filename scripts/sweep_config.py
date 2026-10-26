"""Expand the retained training configurations without starting training."""
from __future__ import annotations
import datetime as dt

DEFAULT_NAMING = {
    "run_name": "{method}_{taskset}_seed{seed}",
    "wandb_group": "{method}_{taskset}",
    "wandb_tags": ["{method}", "{taskset}", "seed{seed}", "{date}", "{sweep_id}"],
    "logdir_folder": "{taskset}/seed_{seed}",
}

def apply_pretrained_encoder(jobs: list[dict], path: str, freeze: bool) -> list[dict]:
    """Append pretrained-encoder flags to each job + add a name/logdir suffix.

    Sequential (argparse) jobs get `--pretrained-encoder PATH [--freeze-encoder
    | --no-freeze-encoder]`; single (Hydra) jobs get `pretrained_encoder=PATH
    freeze_encoder=<bool>`. The `_preEnc`/`_preEncFr` suffix keeps these runs
    separate from the from-scratch baseline in wandb + on disk.
    """
    suffix = "_preEncFr" if freeze else "_preEnc"
    tag = suffix[1:]
    for j in jobs:
        ea = list(j.get("extra_args", []))
        if j.get("mode") == "single":
            ea += [f"pretrained_encoder={path}",
                   f"freeze_encoder={'true' if freeze else 'false'}"]
        else:
            ea += ["--pretrained-encoder", str(path),
                   "--freeze-encoder" if freeze else "--no-freeze-encoder"]
        j["extra_args"] = ea
        j["run_name"] = j["run_name"] + suffix
        j["wandb_group"] = j["wandb_group"] + suffix
        j["logdir"] = j["logdir"] + suffix
        j["wandb_tags"] = list(j.get("wandb_tags", [])) + [tag]
    return jobs


def parse_filters(filter_strs: list[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for s in filter_strs:
        for pair in s.split(","):
            pair = pair.strip()
            if not pair:
                continue
            k, _, v = pair.partition("=")
            out.setdefault(k.strip(), set()).add(v.strip())
    return out


def _fmt(template: str, ctx: dict) -> str:
    try:
        return template.format(**ctx)
    except KeyError as e:
        raise KeyError(
            f"Naming template {template!r} references unknown placeholder {e}. "
            f"Available: {sorted(ctx)}"
        ) from None


def expand_jobs(cfg: dict, *, seeds_override: list[int] | None) -> list[dict]:
    seeds = seeds_override if seeds_override is not None else cfg["seeds"]
    sweep_id = cfg["sweep_id"]
    today = dt.date.today().isoformat()
    logdir_root = cfg["logdir_root"]
    wandb_entity = cfg.get("wandb_entity")
    naming = {**DEFAULT_NAMING, **(cfg.get("naming") or {})}
    defaults = cfg.get("defaults") or {}

    jobs: list[dict] = []
    for cell in cfg["matrix"]:
        method = cell["method"]
        method_cfg = cfg["methods"][method]
        for taskset in cell["tasksets"]:
            ts_cfg = cfg["tasksets"][taskset]
            params = {**defaults, **ts_cfg}

            for seed in seeds:
                ctx = {
                    "method": method, "taskset": taskset, "seed": seed,
                    "sweep_id": sweep_id, "date": today,
                }
                run_name = _fmt(naming["run_name"], ctx)
                wandb_group = _fmt(naming["wandb_group"], ctx)
                wandb_tags = [_fmt(t, ctx) for t in naming["wandb_tags"]]
                logdir_folder = _fmt(naming["logdir_folder"], ctx)
                logdir_subdir = method_cfg["logdir_subdir"]
                logdir = f"{logdir_root}/{logdir_subdir}/{logdir_folder}"

                jobs.append({
                    "mode": "single",
                    "method": method,
                    "taskset": taskset,
                    "seed": seed,
                    "run_name": run_name,
                    "wandb_entity": wandb_entity,
                    "wandb_project": method_cfg["wandb_project"],
                    "wandb_group": wandb_group,
                    "wandb_tags": wandb_tags,
                    "entry": method_cfg["entry"],
                    "logdir": logdir,
                    # TD-MPC2-specific knobs:
                    "task":              params["task"],
                    "steps":             params["steps"],
                    "buffer_size":       params["buffer_size"],
                    "model_size":        params["model_size"],
                    "image_size":        params["image_size"],
                    "max_episode_steps": params["max_episode_steps"],
                    "action_repeat":     params["action_repeat"],
                    "cameras":           params["cameras"],
                    "batch_size":        params["batch_size"],
                    "horizon":           params["horizon"],
                    "mpc":               params["mpc"],
                    "eval_freq":         params["eval_freq"],
                    "eval_episodes":     params["eval_episodes"],
                    "episodic":          params["episodic"],
                    # Replay-buffer storage.
                    "buffer_storage_device":  params.get("buffer_storage_device", "auto"),
                    # Free-form Hydra overrides appended to the `python train.py …`
                    # invocation (e.g. PRISM-WM: use_moe=true use_orthogonal=true …).
                    "extra_args":             list(method_cfg.get("extra_args", [])),
                })
    return jobs


def _is_sequential_schema(cfg: dict) -> bool:
    """A YAML is sequential if any taskset entry has a `tasks: [...]` list."""
    tasksets = cfg.get("tasksets") or {}
    for v in tasksets.values():
        if isinstance(v, dict) and isinstance(v.get("tasks"), list):
            return True
    return False


def expand_sequential_jobs(cfg: dict, *, seeds_override: list[int] | None) -> list[dict]:
    """Expand a sequential-schema YAML.

    Per-taskset:    `tasks: [env1, env2, ...]` (the curriculum).
    Per-method:     `methods.<m>.tasksets.<ts> = { steps: [...], buffer_size_divisor, ... }`.
                    Optional `buffer_sizes: [...]` overrides the divisor.
    Each job carries lists for tasks/task_steps/task_buffer_sizes.
    """
    seeds = seeds_override if seeds_override is not None else cfg["seeds"]
    sweep_id = cfg["sweep_id"]
    today = dt.date.today().isoformat()
    logdir_root = cfg["logdir_root"]
    wandb_entity = cfg.get("wandb_entity")
    naming = {**DEFAULT_NAMING, **(cfg.get("naming") or {})}
    defaults = cfg.get("defaults") or {}

    jobs: list[dict] = []
    for cell in cfg["matrix"]:
        method = cell["method"]
        method_cfg = cfg["methods"][method]
        method_tasksets = method_cfg.get("tasksets") or {}
        for taskset in cell["tasksets"]:
            ts_def = cfg["tasksets"][taskset]
            tasks_list = list(ts_def["tasks"])
            ts_method_cfg = method_tasksets.get(taskset, {})
            steps = list(ts_method_cfg["steps"])
            if len(steps) != len(tasks_list):
                raise ValueError(
                    f"{method}/{taskset}: {len(tasks_list)} tasks but "
                    f"{len(steps)} steps."
                )
            if "buffer_sizes" in ts_method_cfg:
                buffer_sizes = list(ts_method_cfg["buffer_sizes"])
                if len(buffer_sizes) != len(tasks_list):
                    raise ValueError(
                        f"{method}/{taskset}: buffer_sizes length "
                        f"{len(buffer_sizes)} != tasks length {len(tasks_list)}"
                    )
            else:
                divisor = int(
                    ts_method_cfg.get("buffer_size_divisor")
                    or ts_def.get("buffer_size_divisor")
                    or defaults.get("buffer_size_divisor", 8)
                )
                buffer_sizes = [s // divisor for s in steps]

            params = {**defaults, **{
                k: v for k, v in ts_def.items() if k != "tasks"
            }, **{
                k: v for k, v in ts_method_cfg.items()
                if k not in ("steps", "buffer_sizes", "buffer_size_divisor")
            }}

            for seed in seeds:
                ctx = {
                    "method": method, "taskset": taskset, "seed": seed,
                    "sweep_id": sweep_id, "date": today,
                }
                run_name = _fmt(naming["run_name"], ctx)
                wandb_group = _fmt(naming["wandb_group"], ctx)
                wandb_tags = [_fmt(t, ctx) for t in naming["wandb_tags"]]
                logdir_folder = _fmt(naming["logdir_folder"], ctx)
                logdir_subdir = method_cfg["logdir_subdir"]
                logdir = f"{logdir_root}/{logdir_subdir}/{logdir_folder}"

                jobs.append({
                    "mode": "sequential",
                    "method": method,
                    "taskset": taskset,
                    "seed": seed,
                    "run_name": run_name,
                    "wandb_entity": wandb_entity,
                    "wandb_project": method_cfg["wandb_project"],
                    "wandb_group": wandb_group,
                    "wandb_tags": wandb_tags,
                    "entry": method_cfg["entry"],
                    "logdir": logdir,
                    # Sequential-specific (lists of length len(tasks)).
                    "tasks":              tasks_list,
                    "task_steps":         [int(s) for s in steps],
                    "task_buffer_sizes":  [int(b) for b in buffer_sizes],
                    "extra_args":         list(method_cfg.get("extra_args", [])),
                    # Shared TD-MPC2 knobs (scalars).
                    "model_size":        params["model_size"],
                    "image_size":        params["image_size"],
                    "max_episode_steps": params["max_episode_steps"],
                    "action_repeat":     params["action_repeat"],
                    "cameras":           params["cameras"],
                    "batch_size":        params["batch_size"],
                    "horizon":           params["horizon"],
                    "mpc":               params["mpc"],
                    "eval_freq":         params["eval_freq"],
                    "eval_episodes":     params["eval_episodes"],
                    "episodic":          params["episodic"],
                })
    return jobs


def apply_filters(jobs: list[dict], filt: dict[str, set[str]]) -> list[dict]:
    if not filt:
        return jobs
    out = []
    for j in jobs:
        ok = True
        for k, vs in filt.items():
            jv = str(j.get(k))
            if jv not in vs:
                ok = False
                break
        if ok:
            out.append(j)
    return out


def expand_dreamer_jobs(cfg: dict, *, seeds_override: list[int] | None) -> list[dict]:
    seeds = seeds_override if seeds_override is not None else cfg["seeds"]
    sweep_id = cfg["sweep_id"]
    today = dt.date.today().isoformat()
    logdir_root = cfg["logdir_root"]
    wandb_entity = cfg.get("wandb_entity")
    naming = {**DEFAULT_NAMING, **(cfg.get("naming") or {})}

    jobs: list[dict] = []
    for cell in cfg["matrix"]:
        method = cell["method"]
        method_cfg = cfg["methods"][method]
        for taskset in cell["tasksets"]:
            taskset_tasks = cfg["tasksets"][taskset]["tasks"]
            ts_cfg = method_cfg["tasksets"][taskset]
            steps = ts_cfg["steps"]
            configs = ts_cfg["configs"]
            divisor = ts_cfg["dataset_sizes_divisor"]
            dataset_sizes = [s // divisor for s in steps]
            assert len(taskset_tasks) == len(steps) == len(configs), (
                f"Mismatched tasks/configs/steps for {method}/{taskset}: "
                f"{len(taskset_tasks)} tasks, {len(configs)} configs, {len(steps)} steps"
            )

            for seed in seeds:
                ctx = {
                    "method": method, "taskset": taskset, "seed": seed,
                    "sweep_id": sweep_id, "date": today,
                }
                run_name = _fmt(naming["run_name"], ctx)
                wandb_group = _fmt(naming["wandb_group"], ctx)
                wandb_tags = [_fmt(t, ctx) for t in naming["wandb_tags"]]
                logdir_folder = _fmt(naming["logdir_folder"], ctx)
                logdir_subdir = method_cfg["logdir_subdir"]
                logdir = f"{logdir_root}/{logdir_subdir}/{logdir_folder}"

                jobs.append({
                    "method": method,
                    "taskset": taskset,
                    "seed": seed,
                    "run_name": run_name,
                    "wandb_entity": wandb_entity,
                    "wandb_project": method_cfg["wandb_project"],
                    "wandb_group": wandb_group,
                    "wandb_tags": wandb_tags,
                    "entry": method_cfg["entry"],
                    "tasks": taskset_tasks,
                    "configs": configs,
                    "steps": steps,
                    "dataset_sizes": dataset_sizes,
                    "logdir": logdir,
                    "extra_args": method_cfg.get("extra_args", []),
                })
    return jobs
