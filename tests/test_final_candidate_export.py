"""Tests for freezing, parity verification, contract export, and model card (Phase 9.8)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from war_game_engine.contracts import ContractEnvelope, parse_contract_json

from war_ai.rl.final_candidate_export import (
    CorruptedCandidateArtifactError,
    IncompatibleCandidateRulesetError,
    freeze_and_export_final_candidate,
    reload_frozen_candidate_runner,
    verify_candidate_rules_compatibility,
    verify_frozen_candidate_integrity,
)


def test_freeze_and_export_final_candidate_parity_and_model_card() -> None:
    bundle = freeze_and_export_final_candidate(seed=980)

    assert bundle.frozen_before_final_tests
    assert bundle.weights_sha256.startswith("sha256:")
    assert bundle.parity_report.passed
    assert bundle.parity_report.logits_max_abs_diff == 0.0
    assert bundle.parity_report.values_max_abs_diff == 0.0
    assert bundle.parity_report.belief_logits_max_abs_diff == 0.0
    assert bundle.parity_report.action_selection_parity
    assert bundle.parity_report.recurrent_state_isolation_verified
    assert bundle.parity_report.contract_envelopes_valid
    assert bundle.parity_report.rules_compatibility_verified

    # Verify PolicyManifest contract round-trip
    env = ContractEnvelope(
        schema_version="1.0",
        api_version="1.0.0",
        payload=bundle.policy_manifest,
    )
    assert parse_contract_json(env.model_dump_json()).payload == bundle.policy_manifest

    # Verify RecurrentStateExportSpec memory initialization
    mem = bundle.recurrent_state_spec.create_memory(num_games=2)
    assert mem is not None

    # Verify Model Card Markdown contains all required sections
    md = bundle.model_card_markdown
    assert "# Model Card" in md
    assert bundle.weights_sha256 in md
    assert "Plateau local" in md
    assert "não** prova ótimo global" in md

    # Verify runner reload works
    runner = reload_frozen_candidate_runner(bundle, profile_id="fast", seed=980)
    assert runner.profile.profile_id == "fast"


def test_corrupted_weights_and_incompatible_rulesets_are_rejected() -> None:
    bundle = freeze_and_export_final_candidate(seed=981)
    verify_frozen_candidate_integrity(bundle)
    verify_candidate_rules_compatibility(bundle)

    # Corrupt 1 byte of weights
    tampered_bytes = bytearray(bundle.weights_bytes)
    tampered_bytes[0] ^= 0xFF
    corrupted_bundle = replace(bundle, weights_bytes=bytes(tampered_bytes))
    with pytest.raises(CorruptedCandidateArtifactError):
        verify_frozen_candidate_integrity(corrupted_bundle)

    # Incompatible ruleset or version
    with pytest.raises(IncompatibleCandidateRulesetError):
        verify_candidate_rules_compatibility(bundle, ruleset_id="war2-modern")
    with pytest.raises(IncompatibleCandidateRulesetError):
        verify_candidate_rules_compatibility(bundle, rules_version="2.0.0")


def test_persisted_phase9_final_candidate_benchmark_artifact() -> None:
    import json
    from pathlib import Path

    artifact_path = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "phase-9.8-final-candidate-2026-10-28.json"
    )
    assert artifact_path.is_file()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "9"
    assert payload["belief_comparison"]["all_respect_hard_constraints"] is True
    assert payload["information_safety_audit"]["zero_information_leakage"] is True
    assert payload["combat_dp_comparison"]["semantically_equivalent"] is True
    assert (
        payload["execution_profiles"]["all_profiles_evaluated_separately"] is True
    )
    export_data = payload["final_candidate_export"]
    assert export_data["frozen_before_final_tests"] is True
    assert export_data["parity_verification"]["passed"] is True
    assert export_data["evidence_selection"]["local_plateau_reached"] is True
    assert export_data["evidence_selection"]["proves_global_optimum"] is False

