import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"
os.environ["EGL_LOG_LEVEL"] = "fatal"
import warnings
warnings.filterwarnings("ignore", message="Constant.*may be too high")
warnings.filterwarnings("ignore", message=".*Please upgrade to Gymnasium.*")
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
# import warnings
warnings.filterwarnings('ignore')
import shutil
from pathlib import Path

import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def begin_run(work_dir):
	"""Start from the beginning: delete what an interrupted attempt left in
	`work_dir`. Nothing is saved during training (the agent is written to
	`models/final.pt` when it ends), so a rerun trains as an uninterrupted run
	would, as DreamerV3 does. A finished run (`models/final.pt` exists) is left
	untouched and False is returned."""
	work_dir = Path(work_dir)
	if (work_dir / 'models' / 'final.pt').exists():
		return False
	for name in ('models', 'eval_video', 'train_eps'):
		shutil.rmtree(work_dir / name, ignore_errors=True)
	for name in ('eval.csv', 'manifest.json'):
		(work_dir / name).unlink(missing_ok=True)
	return True


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	if not begin_run(cfg.work_dir):
		print(colored(f'Already finished: {cfg.work_dir}/models/final.pt. Nothing to do.', 'green'))
		return
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
	trainer = trainer_cls(
		cfg=cfg,
		env=make_env(cfg),
		agent=TDMPC2(cfg),
		buffer=Buffer(cfg),
		logger=Logger(cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
