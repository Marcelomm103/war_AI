"""Exact combat probability distributions and DP campaign evaluation (Phase 9.4)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Literal, TypeAlias

from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.combat import (
    CombatDiceFaces,
    enumerate_combat_loss_distribution,
    resolve_combat_round,
)
from war_game_engine.contracts import (
    Action,
    AttackOperation,
    ChanceOutcome,
    FullGameState,
    OccupyTerritoryOperation,
)
from war_game_engine.engine import (
    apply_action,
    apply_combat_outcome,
)

CombatActionChoice: TypeAlias = Literal[
    "stop",
    "roll_1",
    "roll_2",
    "roll_3",
]


class UnsafeMacroSubstitutionError(ValueError):
    """Raised when a multi-round combat macro ignores stop, occupation, or victory."""


@dataclass(frozen=True)
class CertifiedCombatLossBranch:
    """Single exact loss outcome branch with certified probability and canonical faces."""

    attacker_dice_count: int
    defender_dice_count: int
    attacker_losses: int
    defender_losses: int
    exact_probability: Fraction
    probability: float
    canonical_faces: CombatDiceFaces


def _build_certified_combat_catalog() -> (
    dict[tuple[int, int], tuple[CertifiedCombatLossBranch, ...]]
):
    catalog: dict[tuple[int, int], tuple[CertifiedCombatLossBranch, ...]] = {}
    for a_dice in (1, 2, 3):
        for d_dice in (1, 2, 3):
            dist = enumerate_combat_loss_distribution(a_dice, d_dice)
            representative_faces: dict[tuple[int, int], CombatDiceFaces] = {}
            for roll in product(range(1, 7), repeat=a_dice + d_dice):
                a_faces = tuple(roll[:a_dice])
                d_faces = tuple(roll[a_dice:])
                losses = resolve_combat_round(a_faces, d_faces)
                if losses not in representative_faces:
                    representative_faces[losses] = CombatDiceFaces(
                        attacker_faces=a_faces,
                        defender_faces=d_faces,
                    )
            branches = tuple(
                CertifiedCombatLossBranch(
                    attacker_dice_count=a_dice,
                    defender_dice_count=d_dice,
                    attacker_losses=a_loss,
                    defender_losses=d_loss,
                    exact_probability=frac,
                    probability=float(frac),
                    canonical_faces=representative_faces[(a_loss, d_loss)],
                )
                for (a_loss, d_loss), frac in sorted(dist.items())
            )
            catalog[(a_dice, d_dice)] = branches
    return catalog


CERTIFIED_COMBAT_CATALOG: Mapping[
    tuple[int, int],
    tuple[CertifiedCombatLossBranch, ...],
] = _build_certified_combat_catalog()


def get_certified_combat_branches(
    attacker_dice_count: int,
    defender_dice_count: int,
) -> tuple[CertifiedCombatLossBranch, ...]:
    """Return certified exact loss branches for (attacker_dice_count, defender_dice_count)."""
    key = (attacker_dice_count, defender_dice_count)
    if key not in CERTIFIED_COMBAT_CATALOG:
        raise ValueError(
            "Combat dice counts must be in {1, 2, 3} for both attacker and defender."
        )
    return CERTIFIED_COMBAT_CATALOG[key]


@dataclass(frozen=True)
class CampaignMacroSpec:
    """Specification of a multi-round campaign evaluation or macro."""

    allow_optional_stop: bool = True
    model_mandatory_occupation: bool = True
    check_intermediate_victory: bool = True


def validate_campaign_macro_equivalence(spec: CampaignMacroSpec) -> None:
    """Reject any multi-round macro that ignores stopping, occupation, or victory."""
    if not spec.allow_optional_stop:
        raise UnsafeMacroSubstitutionError(
            "Multi-round combat macros must not ignore intermediate stop decisions."
        )
    if not spec.model_mandatory_occupation:
        raise UnsafeMacroSubstitutionError(
            "Multi-round combat macros must not ignore mandatory post-conquest occupation."
        )
    if not spec.check_intermediate_victory:
        raise UnsafeMacroSubstitutionError(
            "Multi-round combat macros must not ignore intermediate objective victory checks."
        )


@dataclass(frozen=True)
class CampaignDPStateResult:
    """Exact DP solution for an (attacker_armies, defender_armies) state."""

    attacker_armies: int
    defender_armies: int
    optimal_choice: CombatActionChoice
    optimal_dice_count: int | None
    expected_value: float
    conquest_probability: float
    immediate_victory_probability: float
    expected_Remaining_origin_armies: float
    expected_occupied_target_armies: float
    optimal_occupation_by_remaining_armies: Mapping[tuple[int, int], int]


@dataclass(frozen=True)
class AggregatedVsRoundByRoundComparison:
    """Comparison report between exact DP / aggregated search and round-by-round tree."""

    initial_attacker_armies: int
    initial_defender_armies: int
    dp_optimal_choice: CombatActionChoice
    round_by_round_optimal_choice: CombatActionChoice
    dp_expected_value: float
    round_by_round_expected_value: float
    max_abs_value_difference: float
    dp_conquest_probability: float
    round_by_round_conquest_probability: float
    semantically_equivalent: bool
    dp_states_evaluated: int
    round_by_round_tree_leaves: int
    naive_unsafe_macro_value_bias: float


def default_campaign_terminal_evaluator(
    *,
    origin_armies: int,
    target_armies: int,
    conquered: bool,
    occupied_armies: int,
    requires_two_armies_per_territory: bool = False,
    conquest_triggers_victory: bool = False,
    two_army_conquest_triggers_victory: bool = False,
) -> tuple[float, bool]:
    """Evaluate terminal campaign outcome and whether immediate victory occurs."""
    if conquered:
        both_have_two = origin_armies >= 2 and occupied_armies >= 2
        if conquest_triggers_victory or (
            two_army_conquest_triggers_victory and both_have_two
        ):
            return 1.0, True
        base_val = 0.45 + 0.06 * float(origin_armies + occupied_armies)
        if requires_two_armies_per_territory and both_have_two:
            base_val += 0.25
        elif requires_two_armies_per_territory and not both_have_two:
            base_val -= 0.15
        return min(0.95, base_val), False

    # Stopped before conquering target
    stop_val = 0.05 * float(origin_armies) - 0.07 * float(target_armies)
    if requires_two_armies_per_territory and origin_armies >= 2:
        stop_val += 0.08
    return max(-0.95, min(0.85, stop_val)), False


def evaluate_territory_campaign_dp(
    initial_attacker_armies: int,
    initial_defender_armies: int,
    *,
    macro_spec: CampaignMacroSpec = CampaignMacroSpec(),
    terminal_evaluator: Callable[..., tuple[float, bool]] = (
        default_campaign_terminal_evaluator
    ),
    requires_two_armies_per_territory: bool = False,
    conquest_triggers_victory: bool = False,
    two_army_conquest_triggers_victory: bool = False,
) -> tuple[CampaignDPStateResult, dict[tuple[int, int], CampaignDPStateResult]]:
    """Solve single-territory combat campaign exactly via bottom-up dynamic programming."""
    validate_campaign_macro_equivalence(macro_spec)
    if isinstance(initial_attacker_armies, bool) or initial_attacker_armies < 1:
        raise ValueError("initial_attacker_armies must be >= 1.")
    if isinstance(initial_defender_armies, bool) or initial_defender_armies < 1:
        raise ValueError("initial_defender_armies must be >= 1.")

    table: dict[tuple[int, int], CampaignDPStateResult] = {}
    optimal_occupation_map: dict[tuple[int, int], int] = {}

    def _conquest_resolution(
        surviving_attacker_armies: int,
        last_attacker_dice: int,
    ) -> tuple[float, float, int, int]:
        max_move = min(last_attacker_dice, surviving_attacker_armies - 1)
        best_val = -1e9
        best_is_win = False
        best_move = 1
        for move_armies in range(1, max_move + 1):
            rem_origin = surviving_attacker_armies - move_armies
            val, is_win = terminal_evaluator(
                origin_armies=rem_origin,
                target_armies=0,
                conquered=True,
                occupied_armies=move_armies,
                requires_two_armies_per_territory=requires_two_armies_per_territory,
                conquest_triggers_victory=conquest_triggers_victory,
                two_army_conquest_triggers_victory=two_army_conquest_triggers_victory,
            )
            if val > best_val:
                best_val = val
                best_is_win = is_win
                best_move = move_armies
        optimal_occupation_map[(surviving_attacker_armies, last_attacker_dice)] = (
            best_move
        )
        return (
            best_val,
            1.0 if best_is_win else 0.0,
            surviving_attacker_armies - best_move,
            best_move,
        )

    for total_sum in range(2, initial_attacker_armies + initial_defender_armies + 1):
        for a_armies in range(1, min(initial_attacker_armies, total_sum - 1) + 1):
            d_armies = total_sum - a_armies
            if d_armies < 1 or d_armies > initial_defender_armies:
                continue

            stop_val, stop_win = terminal_evaluator(
                origin_armies=a_armies,
                target_armies=d_armies,
                conquered=False,
                occupied_armies=0,
                requires_two_armies_per_territory=requires_two_armies_per_territory,
                conquest_triggers_victory=conquest_triggers_victory,
                two_army_conquest_triggers_victory=two_army_conquest_triggers_victory,
            )

            best_choice: CombatActionChoice = "stop"
            best_dice: int | None = None
            best_ev = stop_val
            best_conq_p = 0.0
            best_win_p = 1.0 if stop_win else 0.0
            best_rem_origin = float(a_armies)
            best_occ_target = 0.0

            if a_armies >= 2:
                d_dice = min(3, d_armies)
                for a_dice in range(1, min(3, a_armies - 1) + 1):
                    branches = get_certified_combat_branches(a_dice, d_dice)
                    ev = 0.0
                    conq_p = 0.0
                    win_p = 0.0
                    rem_origin_ev = 0.0
                    occ_target_ev = 0.0

                    for br in branches:
                        next_a = a_armies - br.attacker_losses
                        next_d = d_armies - br.defender_losses
                        prob = br.probability
                        if next_d == 0:
                            c_val, c_win_p, rem_orig, occ_targ = _conquest_resolution(
                                next_a,
                                a_dice,
                            )
                            ev += prob * c_val
                            conq_p += prob * 1.0
                            win_p += prob * c_win_p
                            rem_origin_ev += prob * float(rem_orig)
                            occ_target_ev += prob * float(occ_targ)
                        else:
                            sub = table[(next_a, next_d)]
                            ev += prob * sub.expected_value
                            conq_p += prob * sub.conquest_probability
                            win_p += prob * sub.immediate_victory_probability
                            rem_origin_ev += prob * sub.expected_Remaining_origin_armies
                            occ_target_ev += prob * sub.expected_occupied_target_armies

                    if ev > best_ev + 1e-12:
                        best_ev = ev
                        best_dice = a_dice
                        best_choice = (
                            "roll_1"
                            if a_dice == 1
                            else ("roll_2" if a_dice == 2 else "roll_3")
                        )
                        best_conq_p = conq_p
                        best_win_p = win_p
                        best_rem_origin = rem_origin_ev
                        best_occ_target = occ_target_ev

            table[(a_armies, d_armies)] = CampaignDPStateResult(
                attacker_armies=a_armies,
                defender_armies=d_armies,
                optimal_choice=best_choice,
                optimal_dice_count=best_dice,
                expected_value=best_ev,
                conquest_probability=best_conq_p,
                immediate_victory_probability=best_win_p,
                expected_Remaining_origin_armies=best_rem_origin,
                expected_occupied_target_armies=best_occ_target,
                optimal_occupation_by_remaining_armies=dict(optimal_occupation_map),
            )

    root_res = table[(initial_attacker_armies, initial_defender_armies)]
    return root_res, table


def evaluate_round_by_round_exact_tree(
    initial_attacker_armies: int,
    initial_defender_armies: int,
    *,
    terminal_evaluator: Callable[..., tuple[float, bool]] = (
        default_campaign_terminal_evaluator
    ),
    requires_two_armies_per_territory: bool = False,
    conquest_triggers_victory: bool = False,
    two_army_conquest_triggers_victory: bool = False,
) -> tuple[CombatActionChoice, float, float, int]:
    """Unfold the explicit round-by-round attack/chance/occupation tree without memoization."""
    leaf_counter = 0

    def _recurse(a_armies: int, d_armies: int) -> tuple[CombatActionChoice, float, float]:
        nonlocal leaf_counter
        stop_val, _ = terminal_evaluator(
            origin_armies=a_armies,
            target_armies=d_armies,
            conquered=False,
            occupied_armies=0,
            requires_two_armies_per_territory=requires_two_armies_per_territory,
            conquest_triggers_victory=conquest_triggers_victory,
            two_army_conquest_triggers_victory=two_army_conquest_triggers_victory,
        )
        best_choice: CombatActionChoice = "stop"
        best_ev = stop_val
        best_conq_p = 0.0

        if a_armies < 2:
            leaf_counter += 1
            return best_choice, best_ev, best_conq_p

        d_dice = min(3, d_armies)
        for a_dice in range(1, min(3, a_armies - 1) + 1):
            ev = 0.0
            conq_p = 0.0
            for br in get_certified_combat_branches(a_dice, d_dice):
                next_a = a_armies - br.attacker_losses
                next_d = d_armies - br.defender_losses
                if next_d == 0:
                    max_move = min(a_dice, next_a - 1)
                    best_occ_val = -1e9
                    for move_armies in range(1, max_move + 1):
                        leaf_counter += 1
                        val, _ = terminal_evaluator(
                            origin_armies=next_a - move_armies,
                            target_armies=0,
                            conquered=True,
                            occupied_armies=move_armies,
                            requires_two_armies_per_territory=(
                                requires_two_armies_per_territory
                            ),
                            conquest_triggers_victory=conquest_triggers_victory,
                            two_army_conquest_triggers_victory=(
                                two_army_conquest_triggers_victory
                            ),
                        )
                        if val > best_occ_val:
                            best_occ_val = val
                    ev += br.probability * best_occ_val
                    conq_p += br.probability
                else:
                    _, sub_ev, sub_conq = _recurse(next_a, next_d)
                    ev += br.probability * sub_ev
                    conq_p += br.probability * sub_conq

            if ev > best_ev + 1e-12:
                best_ev = ev
                best_conq_p = conq_p
                best_choice = (
                    "roll_1"
                    if a_dice == 1
                    else ("roll_2" if a_dice == 2 else "roll_3")
                )
        return best_choice, best_ev, best_conq_p

    choice, ev, conq_p = _recurse(initial_attacker_armies, initial_defender_armies)
    return choice, ev, conq_p, leaf_counter


def _evaluate_unsafe_naive_macro(
    initial_attacker_armies: int,
    initial_defender_armies: int,
    *,
    terminal_evaluator: Callable[..., tuple[float, bool]] = (
        default_campaign_terminal_evaluator
    ),
    requires_two_armies_per_territory: bool = False,
    conquest_triggers_victory: bool = False,
    two_army_conquest_triggers_victory: bool = False,
) -> float:
    """Compute value of a naive 'fight-to-the-bitter-end with fixed 1-army occupation' macro."""
    memo: dict[tuple[int, int], float] = {}

    def _solve(a_armies: int, d_armies: int) -> float:
        if (a_armies, d_armies) in memo:
            return memo[(a_armies, d_armies)]
        if a_armies < 2:
            val, _ = terminal_evaluator(
                origin_armies=a_armies,
                target_armies=d_armies,
                conquered=False,
                occupied_armies=0,
                requires_two_armies_per_territory=requires_two_armies_per_territory,
                conquest_triggers_victory=conquest_triggers_victory,
                two_army_conquest_triggers_victory=two_army_conquest_triggers_victory,
            )
            memo[(a_armies, d_armies)] = val
            return val
        a_dice = min(3, a_armies - 1)
        d_dice = min(3, d_armies)
        ev = 0.0
        for br in get_certified_combat_branches(a_dice, d_dice):
            next_a = a_armies - br.attacker_losses
            next_d = d_armies - br.defender_losses
            if next_d == 0:
                # Naive macro always moves 1 army and ignores optimal occupation / stop
                val, _ = terminal_evaluator(
                    origin_armies=next_a - 1,
                    target_armies=0,
                    conquered=True,
                    occupied_armies=1,
                    requires_two_armies_per_territory=requires_two_armies_per_territory,
                    conquest_triggers_victory=False,
                    two_army_conquest_triggers_victory=False,
                )
                ev += br.probability * val
            else:
                ev += br.probability * _solve(next_a, next_d)
        memo[(a_armies, d_armies)] = ev
        return ev

    return _solve(initial_attacker_armies, initial_defender_armies)


def compare_aggregated_dp_vs_round_by_round(
    initial_attacker_armies: int = 5,
    initial_defender_armies: int = 3,
    *,
    requires_two_armies_per_territory: bool = True,
    two_army_conquest_triggers_victory: bool = True,
) -> AggregatedVsRoundByRoundComparison:
    """Compare exact DP campaign evaluation against explicit round-by-round expansion."""
    dp_res, dp_table = evaluate_territory_campaign_dp(
        initial_attacker_armies,
        initial_defender_armies,
        requires_two_armies_per_territory=requires_two_armies_per_territory,
        two_army_conquest_triggers_victory=two_army_conquest_triggers_victory,
    )
    rbr_choice, rbr_ev, rbr_conq_p, tree_leaves = evaluate_round_by_round_exact_tree(
        initial_attacker_armies,
        initial_defender_armies,
        requires_two_armies_per_territory=requires_two_armies_per_territory,
        two_army_conquest_triggers_victory=two_army_conquest_triggers_victory,
    )
    naive_macro_ev = _evaluate_unsafe_naive_macro(
        initial_attacker_armies,
        initial_defender_armies,
        requires_two_armies_per_territory=requires_two_armies_per_territory,
        two_army_conquest_triggers_victory=two_army_conquest_triggers_victory,
    )
    diff = abs(dp_res.expected_value - rbr_ev)
    return AggregatedVsRoundByRoundComparison(
        initial_attacker_armies=initial_attacker_armies,
        initial_defender_armies=initial_defender_armies,
        dp_optimal_choice=dp_res.optimal_choice,
        round_by_round_optimal_choice=rbr_choice,
        dp_expected_value=dp_res.expected_value,
        round_by_round_expected_value=rbr_ev,
        max_abs_value_difference=diff,
        dp_conquest_probability=dp_res.conquest_probability,
        round_by_round_conquest_probability=rbr_conq_p,
        semantically_equivalent=(
            diff < 1e-12 and dp_res.optimal_choice == rbr_choice
        ),
        dp_states_evaluated=len(dp_table),
        round_by_round_tree_leaves=tree_leaves,
        naive_unsafe_macro_value_bias=abs(dp_res.expected_value - naive_macro_ev),
    )


def evaluate_attack_action_with_exact_combat_distribution(
    state: FullGameState,
    attack_action: Action,
    value_fn: Callable[[FullGameState], float],
) -> float:
    """Evaluate a single AttackOperation by branching over all certified exact dice outcomes."""
    if not isinstance(attack_action.operation, AttackOperation):
        raise ValueError("Expected an AttackOperation action.")
    combat_state = apply_action(state, attack_action, board=WAR1_BOARD_SPEC)
    pending = combat_state.pending_combat
    if pending is None:
        return value_fn(combat_state)

    branches = get_certified_combat_branches(
        pending.attacker_dice_count,
        pending.defender_dice_count,
    )
    expected_val = 0.0
    for idx, br in enumerate(branches):
        outcome = ChanceOutcome(
            contract_type="chance_outcome",
            request_id=pending.request_id,
            event_id=f"{pending.request_id}:exact:{idx}",
            outcome_source="site",
            visibility="public",
            outcome=br.canonical_faces.model_dump(mode="json"),
        )
        post_combat = apply_combat_outcome(
            combat_state,
            outcome,
            board=WAR1_BOARD_SPEC,
        )
        if post_combat.phase_id == "occupation" and post_combat.pending_occupation is not None:
            occ_pending = post_combat.pending_occupation
            best_occ_v = -1e9
            for move_count in range(1, occ_pending.maximum_army_count + 1):
                occ_act = Action(
                    contract_type="action",
                    action_id=f"exact-occ-{post_combat.state_version}-{move_count}",
                    actor_player_id=occ_pending.attacker_player_id,
                    action_schema_version=attack_action.action_schema_version,
                    expected_state_version=post_combat.state_version,
                    operation=OccupyTerritoryOperation(
                        action_type="occupy_territory",
                        origin_territory_id=occ_pending.origin_territory_id,
                        target_territory_id=occ_pending.target_territory_id,
                        army_count=move_count,
                    ),
                )
                after_occ = apply_action(post_combat, occ_act, board=WAR1_BOARD_SPEC)
                best_occ_v = max(best_occ_v, value_fn(after_occ))
            branch_val = best_occ_v
        else:
            branch_val = value_fn(post_combat)
        expected_val += br.probability * branch_val

    return expected_val


__all__ = [
    "AggregatedVsRoundByRoundComparison",
    "CERTIFIED_COMBAT_CATALOG",
    "CampaignDPStateResult",
    "CampaignMacroSpec",
    "CertifiedCombatLossBranch",
    "CombatActionChoice",
    "UnsafeMacroSubstitutionError",
    "compare_aggregated_dp_vs_round_by_round",
    "default_campaign_terminal_evaluator",
    "evaluate_attack_action_with_exact_combat_distribution",
    "evaluate_round_by_round_exact_tree",
    "evaluate_territory_campaign_dp",
    "get_certified_combat_branches",
    "validate_campaign_macro_equivalence",
]
