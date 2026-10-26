"""A fresh W&B run for a run that is started again after an interruption.

Sequential runs resume at the beginning of their first unfinished task (see
`begin_task` in the training scripts); single-task runs start over (`begin_run`
in TD-MPC2 and PWM's `train.py`). W&B cannot rewind a run: the curves of the
interrupted attempt would stay in it, and the re-trained steps would be dropped,
since W&B only accepts increasing steps. So every restart logs into a new run:

  - the new run gets the old run's name, group, tags and config, and the scalar
    history of the completed tasks (every logged step before the first
    unfinished task starts) is copied into it;
  - the old run is left as it is, moved to the group `<group>_failed` and tagged
    `failed-attempt`, so grouped plots show only the new run.

Media (videos) are not copied. Without W&B access (offline mode, no network),
training goes on in a new, empty run and a warning is printed.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import secrets
from pathlib import Path


def resume_step(logdir) -> int:
    """Global step at which the first unfinished task starts: the end of the last
    completed task in `sequential_progress.json`, 0 if none is completed."""
    try:
        tasks = json.loads((Path(logdir) / 'sequential_progress.json').read_text()).get('tasks', {})
    except (OSError, ValueError):
        return 0
    step, i = 0, 0
    while (tasks.get(str(i)) or {}).get('completed'):
        entry = tasks[str(i)]
        step = int(entry.get('global_step_at_end', entry.get('global_env_step_at_end', step)))
        i += 1
    return step


def restart(job: dict, old_id: str, logdir) -> str:
    """Create the W&B run that continues `old_id` and return its id."""
    new_id = secrets.token_hex(4)
    step = resume_step(logdir)
    note = f'{dt.datetime.now().isoformat(timespec="seconds")} {old_id} -> {new_id} from step {step}'
    with open(Path(logdir) / 'wandb_run_history.txt', 'a') as f:
        f.write(note + '\n')
    if os.environ.get('WANDB_MODE', '').lower() in ('offline', 'disabled', 'dryrun'):
        print(f'# W&B is {os.environ["WANDB_MODE"]}: logging into a new run {new_id} '
              f'without the history of {old_id}', flush=True)
        return new_id
    try:
        import wandb
        api = wandb.Api(timeout=60)
        entity = job.get('wandb_entity') or os.environ.get('WANDB_ENTITY') or api.default_entity
        path = f'{entity}/{job["wandb_project"]}/{old_id}'
        try:
            old = api.run(path)
        except Exception:  # never reached W&B (e.g. died during start-up)
            print(f'# W&B: no run {path}; starting a new run {new_id}', flush=True)
            return new_id

        run = wandb.init(project=job['wandb_project'], entity=entity, id=new_id, name=old.name,
                         group=old.group, tags=list(old.tags), config=dict(old.config),
                         notes=f'Continues {old_id}, whose history up to step {step} is copied here.',
                         dir=str(logdir), resume='never')
        rows = 0
        try:
            if step > 0:
                for row in old.scan_history(page_size=1000):      # in increasing _step order
                    s = row.get('_step')
                    if s is None:
                        continue
                    if s >= step:
                        break
                    data = {k: v for k, v in row.items()
                            if not k.startswith('_') and isinstance(v, (int, float))}
                    if data:
                        run.log(data, step=int(s))
                        rows += 1
        finally:
            run.finish()   # the training process attaches to this run next

        old.group = f'{old.group}_failed' if old.group else 'failed'
        old.tags = sorted(set(old.tags) | {'failed-attempt'})
        old.notes = ((old.notes or '') + f'\nInterrupted; continued in run {new_id}.').strip()
        old.update()
        print(f'# W&B: {path} interrupted -> new run {new_id} with its first {rows} rows '
              f'(steps < {step}); the old run is now in group {old.group}', flush=True)
    except Exception as e:  # noqa: BLE001  (training must not fail on W&B)
        print(f'# W&B: could not prepare the new run ({type(e).__name__}: {e}); '
              f'logging into run {new_id} without the earlier history', flush=True)
    return new_id
