# Migration notes

This copy contains TD-MPC2, PWM, and DreamerV3 single-task/sequential training,
MetaWorld, and PWM expert reuse. The original working files, including local
modifications and ignored configuration files, were the migration sources.
The source workspaces have not been modified.

Historical manifests use `<source-pwm>`, `<source-tdmpc2-dreamer>`, and
`<benchmark>` in place of source-machine paths. Their hashes and source-state
records describe the original migration, not the current cleaned files.

The later-modified box-close environment was selected: 10 July 2026 versus
12 May 2026. Modification times do not establish which historical version
produced a particular run. Reward and observation behavior were not rewritten.

The retained encoder variants for TD-MPC2 and PWM are CNN from scratch and
CNN initialized from a supplied pretrained checkpoint, optionally frozen.
Dreamer's architecture is unchanged. Separate encoder-pretraining scripts,
standalone evaluation/visualization tools, and baseline-specific environment
recipes are omitted; PWM expert reuse is retained.

Training now uses `scripts/launch.py`, with the sweep YAMLs as the source of
experiment settings. Redundant training shell scripts and scheduler launchers
were removed. Personal account defaults and old run IDs are not reused. Task
order, budgets, seeds, and method hyperparameters in the sweep matrices are
preserved. All runtime project paths are derived from the checkout location.

One environment now serves all baselines: `environment.yml` and
`requirements.txt` (conda) and `Dockerfile`. They replace `.devcontainer/`,
the earlier requirement files and the Apptainer helper; see
`environment-validation.md`. No Git repository, submodules, datasets or
experiment logs were created here.

Later configuration changes, made on request:

- Every sweep defaults to all of its tasksets with seeds 0, 1 and 2.
- TD-MPC2 and PWM sweeps list encoder variants (`scratch`, `pretrained` with a
  frozen `checkpoints/encoder_ms19.pt`): pretrained runs are added for TD-MPC2
  single-task and fine-tuning and for PWM single-task and progressive.
- The TD-MPC2 and PWM `pnpboxclose` curriculum now runs
  pick-place-redblock → box-close → compo-pickplace-boxclose, matching
  DreamerV3; each task keeps its own step and buffer budget.
- DreamerV3 ER, EWC and PackNet sweeps include `pnpboxclose`, copied from the
  combined DreamerV3 sweep.
- PWM single-task runs with `compile=false`.

`validation.json` records checks from the initial migration;
`portability-validation.json` records the launcher cleanup checks.

Upstream Meta-World material not needed by the benchmark was later removed:
its documentation, tests, test Docker image, packaging files (`pyproject.toml`,
`setup.py`, `MANIFEST.in`), contribution guide, banner, scripted policies,
evaluation helpers, and the MT/ML multi-task registrations. Only the 18 tasks
named in the sweep configurations remain registered, together with the env
modules and the 51 asset files they load. The Meta-World MIT license moved to
`metaworld/LICENSE`. Unused composition tasks (assembly-disassembly,
coffee-push-button-pull, door-open-door-close, window-open-window-close), the
unregistered `compo_pickplace_boxclose_v2.py`, and a `.bk` file were also
removed; identical copies remain in the source workspaces. Before and after
the removal, each retained task was built through `Meta-World/MT1` with the
three-camera wrapper; the compiled MuJoCo models and seeded rollouts
(proprioception, images, rewards, info) were identical.
