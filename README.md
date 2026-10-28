# war_AI
Repository containing an AI that can play war (risk) boardgame

## Package and optional dependencies

The distribution is named `war-ai`; its Python import name is `war_ai`, while
the repository name remains `war_AI`.

This package requires Python 3.11 or newer. CI and the aggregate development
container use Python 3.11.

Install only the features needed:

```bash
python -m pip install -e ".[test]"
python -m pip install -e ".[quality]"
python -m pip install -e ".[training]"
python -m pip install -e ".[service]"
```

The optional `training` extra pins PyTorch 2.14.1, PettingZoo, Gymnasium, and NumPy.
The `service` extra pins FastAPI and Uvicorn. Direct dependency versions are
recorded in `pyproject.toml`.

## PettingZoo AEC environment

`war_ai.envs.War1AECEnv` wraps the WAR 1 engine as a sequential multi-agent
environment. Install the engine package alongside `war_AI` and its `training`
extra; from this directory in the aggregate repository:

```bash
python -m pip install -e "../war_game_engine" -e ".[training]"
```

Each action is an index into the current player's `PlayerObservation.legal_actions`.
The corresponding `action_mask` has one entry for every legal action, so the
space grows with the current state instead of imposing an arbitrary action cap.
Combat chance outcomes are resolved internally before the next player decision.
`observation` is a structured `War1NumericObservation` with the certified
adjacency graph, continent and ownership features, a six-seat player mask,
reinforcement pools, and the focal player's objective and hand. Logarithmic
troop/resource features are accompanied by big-endian byte records and offsets,
so legal integer quantities are preserved without clipping. Public action and
combat history is serialized separately from private card identifiers; event
times are logical and deterministic. `engine_observation` retains the full
player-scoped engine contract. The layout version is available as
`WAR1_RL_OBSERVATION_ENCODING_VERSION`.
Use `agent_iter()` and submit `None` for agents whose `last()` result is
terminated or truncated, as required by PettingZoo's AEC protocol.

```python
from war_ai.envs import War1AECEnv

env = War1AECEnv(num_players=3)
env.reset(seed=42)
for agent in env.agent_iter():
    observation, reward, terminated, truncated, info = env.last()
    if terminated or truncated:
        env.step(None)
    else:
        env.step(env.action_space(agent).sample(observation["action_mask"]))
```

### Structured action tokens

For autoregressive policies, `War1AECEnv` also exposes
`legal_action_token_values`, `legal_action_token_offsets`, and the conditional
`action_token_mask`. `action_prefix_mask(agent, prefix)` returns the next-token
mask for any valid prefix. The sequence begins with an action-type token,
encodes the operation parameters in schema order, then separates the action ID
and requires an explicit confirmation token. The vocabulary has 268 token IDs,
but sequence length and integer quantities are unbounded by an arbitrary action
cap. `step_tokens()` decodes only a complete sequence matching a currently
legal engine action; partial prefixes never change the game.

The same codec is available as `env.action_codec` and `env.step_tokens()` on
`War1GymEnv`. The discrete-index `step()` interface remains available for
PettingZoo and standard Gymnasium integrations.

### Gymnasium focal adapter

`War1GymEnv` returns control only when the focal player has a decision. Pass an
explicit callable for every other seat; the adapter has no implicit opponent
policy. Each `step()` advances the table until the focal player's next decision
or an episode end. `info["semantic_duration"]` counts live AEC decisions in
that interval, including the focal action but excluding dead-agent cleanup and
automatically resolved combat chance outcomes.

The focal episode is `terminated` when the official game ends or that player is
eliminated, even if the remaining table continues. It is `truncated` only when
an explicit `max_semantic_decisions` budget is reached while the focal player
remains active; time away from the table is not termination.

```python
from war_ai.envs import War1GymEnv


def first_legal(observation, rng):
    del rng
    return 0


env = War1GymEnv(
    focal_agent="player_0",
    opponent_policies={
        "player_1": first_legal,
        "player_2": first_legal,
    },
)
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(0)
```

### Masked losses and recurrent memory

The optional `war_ai.rl` package uses PyTorch from the `training` extra.
`joint_action_log_prob` sums masked conditional log-probabilities over the full
token sequence; `joint_action_entropy` and `joint_action_kl_divergence` use the
same per-prefix masks. Padding is excluded with a token-active mask, while an
active position with no legal token or a sampled masked token raises an error.

`PerPlayerRecurrentMemory` stores GRU hidden states or LSTM hidden/cell states
with separate game and player-seat axes. `select`, `update`, and
`reset_players` keep recurrent state scoped to the acting player rather than
mixing opponents' memories.

`PPOActorCritic` provides a local critic and autoregressive action-token policy.
It accepts fixed-width floating-point features derived from the acting player's
observation; `recurrent="gru"` or `"lstm"` adds state across decisions, while
`"none"` keeps the state encoder feedforward. `sample_action` requests the legal
token mask for each prefix, so it only returns a complete currently legal
action. `compute_masked_ppo_loss` applies the clipped objective and masked
entropy to recorded action tokens. `compute_gae` suppresses bootstrap after
termination, bootstraps truncations, and stops advantage propagation at either
episode boundary.

### Exact finite PPO microtasks

Run the six small learning tasks from the repository root:

```bash
python -m war_ai.rl.small_tasks --seed 2026 --updates 100 --batch-size 32
```

Each task enumerates actions from the WAR engine and uses the structured-action
codec's legal-prefix masks. Attack values use the exact one-round dice-loss
distribution; trade rewards and objective completion are evaluated through
engine transitions. Placement, occupation, and movement use explicit local
micro-objectives. These fixed-state tasks validate PPO learning and exact
action selection, not full-game strength or homologation.

The reproducible CPU run with seeds 2026–2031 learned the exact optimum in all
six tasks:

| Task | Initial expected return | Learned expected return | Exact optimum |
|---|---:|---:|---:|
| Legal placement | 0.661 | 0.9998 | 1.000 |
| Attack or stop | 0.046 | 0.2627 | 0.2627 |
| Occupation | 0.666 | 0.9998 | 1.000 |
| Movement | 0.407 | 0.9997 | 1.000 |
| Trades | 5.976 | 11.9993 | 12.000 |
| Objective completion | 0.468 | 0.9998 | 1.000 |

### Progressive curriculum

`war_ai.rl.curriculum.build_default_curriculum()` defines three training-only
tiers:

1. Tactical microtasks from the exact finite suite above.
2. Contract-valid intermediate states for attack, occupation, movement, trades,
   and objective claims. `War1AECEnv` accepts these through an
   `initial_state_factory`; reset checks the game ID, ruleset, complete map,
   seats, active decision, and restorable random streams.
3. Full WAR games using the ordinary environment reset.

`CurriculumScheduler` advances only after the stage's minimum evaluation sample
and mastery threshold. Opponent pools expand from tactical no-opponent tasks to
random/expansionist/conservative baselines, then add objective, continent,
elimination, and exact-probability strategies. `make_curriculum_gym_env`
constructs the selected intermediate or full-game environment with a
seat-isolated opponent assignment.

Only the full-game scenario ID is homologation-eligible. `FullGameReplayReservoir`
accepts completed official arena wins that start from the full initial placement
state; it rejects truncated matches and all tactical/intermediate stages.
Reservoir entries contain private full replays, so keep the buffer local and
access-controlled.

### Historical self-play

`HistoricalSelfPlayLeague` takes a live `FrozenPolicySpec` and immutable prior
snapshots. Historical entries require a checkpoint SHA-256, generation, and
style label; `promote_current` archives a checkpointed live policy before
installing the newer generation. Campaigns require at least one historical
snapshot and always schedule both a historical opponent and a baseline style.

The default schedule covers 3–6 players and rotates the live policy through
every seat for each table size. Seat rotation varies the live policy's color;
distinct game seeds vary mission assignment, which the arena reports per seat.
Campaigns run through `run_arena`, using its independent per-seat policy
factories and RNGs. Policies receive only the acting player's
`War1AECObservation`; a factory that reuses the same callable across seats is
rejected by the arena.

### Seeded promotion gate

`run_paired_seed_evaluation` runs complete arena matches for each seed, rotating
the candidate through every seat against a fixed baseline. Candidate snapshots
must carry a checkpoint SHA-256, and any truncated/failed match invalidates that
seed evaluation. Results keep candidate and baseline official win rates paired
by seed and record the candidate's mission assignments.

`assess_promotion` rejects fewer than 10 seeds, requires positive win-rate gain
on at least 80% of seeds, and applies a one-sided exact paired sign test
(`p < 0.05` by default). The gate uses official arena outcomes only; it has no
timestep or update-count promotion criterion.

`war_ai.rl.agent` trains a full-game `PPOActorCritic` policy on complete WAR 1
matches, stores its content-addressed weights, optimizer state, resume state,
telemetry journal, and verified `RunManifest` in `ReproducibilityStore`, and
exports a seat-isolated `FrozenPolicySpec` via `export_trained_ppo_policy_spec`.
Run the full-game training and 10-seed promotion campaign from the repository:

```bash
python -m war_ai.rl.agent --output-json lib/war_AI/benchmarks/phase-6.6-promotion-2026-10-27.json
```

In the recorded 10-seed campaign (`seeds=2026..2035`, 3 seat rotations per seed,
30 full games total), the exported checkpoint
`sha256:a1ab317308b3b45de162bdf84dfb439680433c9399e5149b80c5d2b8de33daed`
completed 30/30 matches, achieved positive paired gain on 10/10 seeds (`28/30`
candidate wins, `mean_paired_gain = 0.90`, `median_paired_gain = 1.00`,
one-sided exact sign test `p = 0.0009765625 < 0.05`), and passed `assess_promotion`.

### Architecture and algorithm comparison (Phase 7)

Phase 7 separates all eight experimental axes (`algorithm`, `encoder`, `memory`,
`action_representation`, `curriculum`, `shaping`, `league`, `search`) in
`war_ai.rl.experimental_axes` and enforces identical `ComputeBudget` parity on
single-axis comparisons.

- `war_ai.rl.architectures` implements `MLPObservationEncoder`,
  `MessagePassingGNNEncoder`, `GraphTransformerEncoder`, recurrent `GRU`/`LSTM`
  memory, `CausalEventTransformer`, `ObjectiveAndCardEncoder`,
  `PhaseConditionedActionHead`, and `OpponentBeliefHead` inside
  `WarArchitectureModel`.
- `war_ai.rl.algorithms` provides the six minimum branches (`ppo_feedforward`,
  `ppo_recurrent`, `graph_actor_critic`, `impala_vtrace`,
  `recurrent_off_policy_r2d2`, `policy_value_search`) and rejects continuous or
  cooperative-only methods (`sac_continuous`, `ddpg`, `qmix`).
- `war_ai.rl.centralized_training` compares local vs privileged critics,
  restricts opponent secret labels to training losses, verifies counterfactual
  non-interference on actor decisions, and strips privileged critic parameters
  when exporting production policy artifacts.
- `war_ai.rl.hyperparameter_search` and `war_ai.rl.training_sweeps` define the
  log-scale and discrete hyperparameter exploration grids, training regime
  sweeps, and version-compatibility guards (`VersionedReplayBuffer`,
  `verify_trajectory_batch_compatibility`).
- `war_ai.rl.rigorous_comparison` runs controlled-budget screening, tracks
  learning curves by samples/compute/wall-time, executes single-axis ablations,
  preserves negative findings in `NegativeResultRegistry`, performs open-ended
  scaling studies, and enforces a 20-seed paired arena gate for finalists:

```bash
python -m war_ai.rl.rigorous_comparison --output-json lib/war_AI/benchmarks/phase-7.7-comparison-2026-10-28.json
```

### League, PSRO metagame, exploiters, and frozen benchmark panel (Phase 8)

Phase 8 organizes multi-player league training, empirical metagame analysis,
adversarial best-response exploitation, generalization auditing, and frozen
benchmark panel assembly:

- `war_ai.rl.league_population` maintains the six population roles
  (`main_agent`, `historical_snapshot`, `architecture_seed_leader`,
  `mission_style_specialist`, `weakness_exploiter`, and
  `independent_generalization_model`) with mandatory checkpoint SHA-256 digests,
  parameter fingerprints, and strict lineage isolation between training and
  held-out models.
- `war_ai.rl.full_table_training` samples full multi-player tables (`3..6`
  players, rejecting 2-player duels), combines uniform historical sampling with
  prioritized hard-matchup weights, logs exact candidate/seat/joint lineup
  probabilities, and tracks Shannon entropy and effective support size.
- `war_ai.rl.psro_metagame` records empirical payoffs per canonical multi-player
  lineup, solves replicator-dynamics and finite-population Alpha-Rank
  meta-strategies, computes NashConv/regret, compares multiplayer rankings
  against pairwise Elo, detects intransitive 3-cycles ($A \succ B \succ C \succ A$),
  and trains PSRO best-response oracles against policy mixtures.
- `war_ai.rl.best_response` trains at least three independent best-response
  exploiter approaches across multiple seeds, enforces solo official-victory
  maximization with seat-isolated instances and zero secret sharing, and keeps
  deliberate collusion stress tests strictly separated from the normal
  benchmark panel.
- `war_ai.rl.generalization` evaluates cross-play across seeds and
  architectures, performance against held-out models, worst opponent and worst
  lineup gaps, historical non-regression, mission weaknesses, and seat/color
  adaptation bias.
- `war_ai.rl.frozen_benchmark_panel` forms and validates the frozen benchmark
  panel ($\ge 8$ distinct policies, $\ge 3$ successful approaches, leaders,
  held-out opponents, and strong exploiters, rejecting random/weak fillers and
  near-duplicate checkpoints):

```bash
python -m war_ai.rl.frozen_benchmark_panel --output-json lib/war_AI/benchmarks/phase-8.6-league-panel-2026-10-28.json
```

### Training telemetry and resumable checkpoints

`TrainingRunJournal` writes append-only JSONL records for completed episodes
and PPO updates. `EpisodeAccumulator` reads reset/step info and records official
win reward separately from shaped reward. Curves group by objective and table
size, count truncations explicitly, and exclude truncated episodes from the
official win-rate denominator. Update events include losses, KL, entropy,
gradient norm, throughput, process RSS, and CUDA peak allocation.

`War1AECEnv.state_dict()` and `War1GymEnv.state_dict()` preserve the live game,
replay/event history, per-seat/environment RNGs, and opponent RNGs. Use
`save_ppo_checkpoint`/`restore_ppo_checkpoint` with a `PPOResumeState` containing
rollout, recurrent-memory, curriculum, and journal state; the returned artifact
references can be placed directly in a `RunManifest`. Checkpoints restore model,
optimizer, Python/NumPy/torch RNGs, and named generators after verifying artifact
hashes. Only restore trusted artifacts from the local `ArtifactStore`.

The microtask CLI can also persist update telemetry:

```bash
python -m war_ai.rl.small_tasks --journal-path runs/microtasks.jsonl
```

### Canonical rewards and optional shaping

The focal adapter uses official WAR victory as its only evaluation reward:
`1` for the official winner and `0` otherwise, including elimination before
the table ends. The episodic baseline fixes `EPISODIC_GAMMA` at `1.0`.
`PotentialRewardShaper` is opt-in; its potential uses only the focal player's
objective and public state, is bounded by `0.05`, and is zero for terminated
states. `truncated` transitions retain their next-state potential. The Gym
`info` reports `official_reward` and `shaping_reward` separately for comparison
against the no-shaping baseline.

### Gameplay baselines

`war_ai.baselines` provides `random_legal`, `expansionist`,
`conservative_borders`, `objective_oriented`, `continent_control`,
`opportunistic_elimination`, and `exact_probability_planner`. Each policy
receives the player-scoped observation and its own seeded RNG, and returns an
index into the current legal-action list. The exact planner uses the engine's
enumerated combat-loss distribution. `random_legal` is for sanity checks and
curricula only; it is not a final homologation opponent.

```python
from war_ai.baselines import create_baseline
from war_ai.envs import War1GymEnv

env = War1GymEnv(
    opponent_policies={
        "player_1": create_baseline("objective_oriented"),
        "player_2": create_baseline("conservative_borders"),
    }
)
```

### Reproducible evaluation arena

`war_ai.arena` runs full-table matches from immutable policy specs. Each seat's
factory receives a deterministic factory seed and is called independently;
policies also receive a separate per-seat RNG. Supply a checkpoint SHA-256 when
the policy is backed by a frozen artifact. The arena records match and seat
seeds, artifact IDs, effective missions, decision counts, timing, official
outcomes, and complete engine replays.
Factories must return an independent inference callable/model per seat; reusing
one callable instance is recorded as a policy-factory failure. The checkpoint
digest is caller-supplied metadata—the factory is responsible for loading the
corresponding immutable artifact.

`max_semantic_decisions` is an action-count budget; reaching it returns a
training truncation, never an unofficial winner. `decision_budget_ms` limits
each policy call and records an explicit failure instead of silently choosing a
fallback action. It is a measured, soft budget: synchronous policy calls cannot
be interrupted while they are still running.

Reports include match, seat, and win counts by policy, effective mission, and
table size. Win rates use completed seat appearances only. Full replays contain
private state such as objectives, cards, and random streams; keep them as local,
restricted evaluation artifacts and do not publish or expose them to policies.

```python
from war_ai.arena import ArenaMatchConfig, baseline_policy_spec, run_arena

assignments = {
    "player_0": "exact_probability_planner",
    "player_1": "objective_oriented",
    "player_2": "conservative_borders",
}
policies = {
    policy_id: baseline_policy_spec(policy_id)
    for policy_id in set(assignments.values())
}
report = run_arena(
    [
        ArenaMatchConfig(
            match_id="baseline-seed-42",
            seed=42,
            num_players=3,
            policy_ids_by_player=assignments,
            max_semantic_decisions=100_000,
            decision_budget_ms=1_000,
        )
    ],
    policies,
)
match = report.matches[0]
print(match.status, match.winner_player_id)
print(report.statistics_by_policy_id)
```

## GPU system requirements

CPU training does not require a GPU driver. GPU support is system-specific and
the host GPU driver is not installed or pinned by this Python package. The
default PyPI wheel may add platform-specific CUDA runtime packages. For a
CPU-only install or a different GPU backend, use the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/) and
its corresponding package index. For NVIDIA hardware, select a build
compatible with the installed driver, check the driver with `nvidia-smi`, and
verify the PyTorch build with `torch.cuda.is_available()`.

The manual GPU smoke workflow is
`.github/workflows/gpu.yml`; it requires a self-hosted Linux runner labeled
`gpu` with a driver compatible with the selected PyTorch wheel. Ordinary CI
uses CPU-only imports and does not install the training extra.

## Experiment tracking

Experiments use a self-contained local store; no hosted tracking service is
required. `war_ai.reproducibility.RunManifest` records source commits and dirty
patches, resolved package versions and an optional lockfile hash, Python/OS,
CPU/accelerator and driver details, complete configuration, per-stream seeds,
dataset/weight/normalizer hashes, optimizer and training-state hashes, and
benchmark settings with the inference budget.

`ReproducibilityStore` writes canonical run JSON under `runs/<run_id>.json` and
content-addressed artifacts under `artifacts/sha256/<prefix>/<digest>`.
`ArtifactStore` streams files while hashing them; saving or verifying a run
checks every referenced artifact's SHA-256 and size. Reusing a run ID with
different metadata is rejected. `capture_installed_dependencies()` records
exact installed distribution versions and `capture_platform_snapshot()`
captures Python and OS details; hardware and accelerator driver details are
provided explicitly by the caller.

These manifests are local provenance, not gameplay contracts. Seeds and
training artifacts must never be added to a player's observation or exposed to
an agent at inference time.

Ruff, mypy, and mutmut are pinned in the `quality` extra. Mutation campaigns
will run once the package contains behavior and corresponding tests.
