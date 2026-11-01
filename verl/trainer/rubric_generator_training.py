"""Single-pass GAP/vertical proposal groups with evaluate-before-commit semantics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable
import uuid
from time import perf_counter

import numpy as np

from .rubric_evolution import CRITERION_STATE_DEAD, criterion_polarity, find_unsupported_numeric_claims
from .rubric_gap_evolution import GapExpansionContext, build_gap_proposal_hint
from .rubric_generator_rewards import parse_rg_proposal, score_rg_proposal
from .rubric_generator_meta_batch import audit_proposal_batches
from .rubric_empirical_batch import CandidateJudgeGroup
from .rubric_meta_verifier import CHILD_EVOLUTION, GAP_EXPAND, format_peer_candidates_hint
from .rubric_meta_recovery import UnavailableAudit, exclude_unavailable_groups


@dataclass
class RGContext:
    prompt_key: str
    question: str
    existing_rubric: list[dict[str, Any]]
    responses: list[str]
    indices: list[int]
    operations: dict[str, str]
    budgets: dict[str, int]
    weights: dict[str, float]
    entry: dict[str, str]
    targets: dict[str, Any]


ACCEPTED_KEYS = {"gap": "rubric_gap/accepted_roots", "vertical": "rubric_evolution/accepted_candidates"}


@dataclass
class RGStage:
    """One role's rollouts and parsed/audited proposals, awaiting Judge scores."""
    role: str
    contexts: list[RGContext]
    rollouts: Any
    assessments: list
    n: int
    results: list[dict[str, float]] = field(default_factory=list)


class RubricGeneratorTraining:
    """Dependencies are fixed at construction; only a per-step rollout buffer is mutable.

    Initial rubric generation remains the existing cached bootstrap/reference
    pipeline. Direct RG policy gradients are applied to GAP and vertical edits.
    """
    def __init__(self, *, config: Any, evolver: Any, meta_router: Any,
                 generate: Callable, audit: Callable, decode: Callable,
                 build_vertical_hint: Callable, parent_id: Callable, operation: Callable):
        self.config = config
        self.evolver = evolver
        self.meta_router = meta_router
        self.generate = generate
        self.audit = audit
        self.decode = decode
        self.build_vertical_hint = build_vertical_hint
        self.parent_id = parent_id
        self.operation = operation
        self.rollouts = []
        self.diagnostics = []
        self.run_id = uuid.uuid4().hex[:12]

    def begin_step(self) -> None:
        self.rollouts.clear()
        self.diagnostics.clear()

    def write_diagnostics(self, step: int) -> Path:
        """Driver-only local JSONL, distinct run IDs prevent resume overwrites."""
        directory = Path(self.config.rubric_generator_log_dir) / self.run_id
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"step_{step:08d}.jsonl"
        temporary = destination.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            for record in self.diagnostics), encoding="utf-8")
        temporary.replace(destination)
        return destination

    def run_step(self, batch: Any, timing_raw: dict, *, step: int, vertical: bool) -> dict[str, float]:
        """GAP and (optionally) vertical RG for one step with one shared Judge pass.

        Both context sets are built from the first-pass diagnostic states before
        any commit: newly accepted GAP roots are never same-step vertical targets,
        and because GAP roots are appended, vertical parent indices stay stable.
        Rewards for both roles exist before either role mutates the rubric.
        """
        role_contexts = {"gap": self._gap_contexts(batch)}
        if vertical:
            role_contexts["vertical"] = self._vertical_contexts(batch, step)
        return self._run_roles(role_contexts, timing_raw=timing_raw, step=step)

    def run_gap(self, batch: Any, timing_raw: dict, *, step: int) -> dict[str, float]:
        return self._run_roles({"gap": self._gap_contexts(batch)}, timing_raw=timing_raw, step=step)

    def run_vertical(self, batch: Any, timing_raw: dict, *, step: int) -> dict[str, float]:
        return self._run_roles({"vertical": self._vertical_contexts(batch, step)}, timing_raw=timing_raw, step=step)

    def _gap_contexts(self, batch: Any) -> list[RGContext]:
        contexts = []
        for key, cache in self.evolver.observed_groups().items():
            state = self.evolver.store.get(key)
            remaining = self.config.rubric_gap_max_roots_per_prompt - sum(
                item.get("source") == "gap_expand" for item in state.parent_rubric) if state is not None else 0
            indices = list(cache.get("group_indices", []))
            if remaining <= 0 or not indices:
                continue
            existing = deepcopy(state.build_active_rubric())
            responses = self.decode(batch, indices)
            question = str(cache["prompt_text"])
            budget = min(remaining, self.config.rubric_gap_max_candidates_per_prompt)
            context = GapExpansionContext(key, question, existing, indices, responses)
            contexts.append(RGContext(key, question, existing, responses, indices,
                {"gap": "gap_expand"}, {"gap": budget}, {"gap": self.config.rubric_gap_root_weight},
                {"question": question, "hint": build_gap_proposal_hint(context, max_candidates=budget)}, {}))
        return contexts

    def _vertical_contexts(self, batch: Any, step: int) -> list[RGContext]:
        # select_targets uses the first-pass diagnostic states. Newly accepted
        # GAP roots are context, not recursively selected for this step.
        grouped = {}
        for target in self.evolver.select_targets(step):
            target.group_responses = self.decode(batch, target.group_indices)
            grouped.setdefault(target.prompt_key, []).append(target)
        contexts = []
        for key, targets in grouped.items():
            target_map = {self.parent_id(target): target for target in targets}
            operations = {key: self.operation(target) for key, target in target_map.items()}
            budgets = {key: self.config.rubric_evolution_max_subcriteria if target.state == CRITERION_STATE_DEAD
                else self.config.rubric_evolution_max_refinements for key, target in target_map.items()}
            weights = {key: float(target.parent_item.get("weight", 1.0)) * self.config.rubric_evolution_child_weight_ratio
                for key, target in target_map.items()}
            first = targets[0]
            contexts.append(RGContext(key, first.prompt_text, deepcopy(self.evolver.store.get(key).build_active_rubric()),
                list(first.group_responses), list(first.group_indices), operations, budgets, weights,
                {"question": first.prompt_text, "hint": self.build_vertical_hint(targets)}, target_map))
        return contexts

    def _audit_proposal(self, context: RGContext, proposal: Any, *, role: str, execution_cache: dict) -> None:
        """Single-proposal compatibility entry point; training uses a stage batch."""
        entries, chunks = RubricGeneratorTraining._prepare_proposal_audit(self, context, proposal, role=role)
        operation_type = GAP_EXPAND if role == "gap" else CHILD_EVOLUTION
        texts = audit_proposal_batches([entries], audit=self.audit,
            template_path=self.meta_router.template_for(operation_type),
            max_tokens=self.config.rubric_evolution_verifier_max_tokens)[0]
        RubricGeneratorTraining._apply_proposal_audit(self, context, chunks, texts,
            role=role, proposal=proposal)
        if not proposal.format_valid or proposal.audit_error:
            return
        RubricGeneratorTraining._judge_proposals(self, [context], [proposal], n=1,
            execution_cache=execution_cache)

    def _prepare_proposal_audit(self, context: RGContext, proposal: Any, *, role: str) -> tuple[list, list]:
        if not proposal.format_valid:
            return [], []
        # Numeric provenance is a deterministic part of the existing semantic
        # contract. Do not let a contradictory PASS override this check.
        for record in proposal.candidates:
            if record.candidate is None:
                continue
            parent = context.targets.get(record.target_id)
            support = context.question + ("\n" + str(parent.parent_item.get("description", "")) if parent is not None else "")
            unsupported = find_unsupported_numeric_claims(record.candidate["description"], support)
            if unsupported:
                record.rejection = "unsupported_numeric_claim"
        records = [record for record in proposal.candidates if record.candidate is not None and not record.rejection]
        if not records:
            return [], []
        operation_type = GAP_EXPAND if role == "gap" else CHILD_EVOLUTION
        entries, chunks = [], []
        limit = self.config.rubric_evolution_max_candidates_per_meta_batch
        # Every meta call sees the complete SAME-completion peer list. It never
        # sees another sampled completion as an already-existing criterion.
        peers = [{"candidate_id": f"{record.target_id}_{record.slot}", "candidate": record.candidate,
                  "parent_id": record.target_id} for record in records]
        for start in range(0, len(records), limit):
            chunk = records[start:start + limit]
            audit_records = []
            for record in chunk:
                item = {"candidate_id": f"{record.target_id}_{record.slot}", "candidate": record.candidate}
                if role == "vertical":
                    target = context.targets[record.target_id]
                    item.update(requested_operation=context.operations[record.target_id], parent_state=target.state,
                        parent={**target.parent_item, "parent_id": record.target_id,
                                "polarity": criterion_polarity(target.parent_item)})
                audit_records.append(item)
            hint = self.meta_router.build_batch_hint(operation_type=operation_type,
                existing_rubric=context.existing_rubric, candidate_records=audit_records)
            hint += "\n\n" + format_peer_candidates_hint(peers)
            entries.append({"question": context.question, "hint": hint, "meta_context": {
                "operation_type": operation_type, "existing_rubric": context.existing_rubric,
                "candidates": audit_records, "peer_candidates": peers,
            }})
            chunks.append(chunk)
        return entries, chunks

    def _apply_proposal_audit(self, context: RGContext, chunks: list, texts: list[str], *,
                              role: str, proposal: Any) -> None:
        """Apply meta-verifier verdicts to each candidate record.

        Unavailable/malformed external audits exclude the complete prompt/role
        GRPO group. They are not evidence of poor generator quality.
        """
        operation_type = GAP_EXPAND if role == "gap" else CHILD_EVOLUTION
        if any(isinstance(text, UnavailableAudit) for text in texts):
            proposal.audit_error = "; ".join(str(text) for text in texts if isinstance(text, UnavailableAudit))
            return
        if len(texts) != len(chunks):
            proposal.audit_error = f"Meta-verifier batch count mismatch: expected {len(chunks)}, got {len(texts)}."
            return
        for chunk, text in zip(chunks, texts, strict=True):
            ids = [f"{record.target_id}_{record.slot}" for record in chunk]
            routed, diagnostics = self.meta_router.route_batch_response(text, expected_candidate_ids=ids, operation_type=operation_type)
            if not diagnostics["parse_valid"] or any(diagnostics.get(key) for key in
                    ("missing_candidate_ids", "duplicate_candidate_ids", "unknown_candidate_ids", "operation_mismatch_candidate_ids")):
                proposal.audit_error = f"Invalid meta-verifier routing: {diagnostics}"
                return
            for record, candidate_id in zip(chunk, ids, strict=True):
                record.meta_payload = routed[candidate_id]
                record.semantic = self.meta_router.parse_candidate(record.meta_payload, operation_type=operation_type)
                if record.semantic.get("rejection_type") == "invalid_json":
                    proposal.audit_error = f"Malformed meta-verifier verdict for candidate {candidate_id}."
                    return

    def _judge_proposals(self, contexts: list[RGContext], proposals: list, *, n: int,
                         execution_cache: dict | None = None) -> dict[str, float]:
        """Single-role compatibility entry point; training uses _judge_stages.

        execution_cache is retained for older callers but intentionally unused:
        empirical judgments must never be shared across sampled proposals.
        """
        if len(proposals) != len(contexts) * n:
            raise RuntimeError("Proposal count mismatch")
        role = "vertical" if any(context.targets for context in contexts) else "gap"
        metrics = RubricGeneratorTraining._judge_stages(self, [RGStage(role, contexts, None, list(proposals), n)])
        return {'empirical_requests': metrics['rg/empirical_requests'],
                'empirical_criteria': metrics['rg/empirical_criteria']}

    def _judge_stages(self, stages: list[RGStage]) -> dict[str, float]:
        """Judge every proposal of every role in one batch, all its criteria per answer.

        A Judge group holds at most one completion per role. GAP completion k and
        vertical completion k of the same prompt share the same Solver responses,
        so they share one request per response: the rubric is the GAP columns
        followed by the vertical columns, split back afterwards. The pairing by
        sample index is arbitrary but fixed. Two completions of the same role are
        never judged in one request, and nothing is shared across groups.
        """
        groups, segments, group_roles, group_indices, slots = [], [], [], [], {}
        stage_id = uuid.uuid4().hex
        for stage in stages:
            if len(stage.assessments) != len(stage.contexts) * stage.n:
                raise RuntimeError("Proposal count mismatch")
            for index, proposal in enumerate(stage.assessments):
                context = stage.contexts[index // stage.n]
                if len(context.responses) != len(context.indices):
                    raise RuntimeError("Incomplete Solver response group")
                records = [record for record in proposal.candidates if record.valid]
                slot = (context.prompt_key, index % stage.n)
                group_index = slots.get(slot)
                if group_index is not None and (stage.role in group_roles[group_index]
                        or groups[group_index].question != context.question
                        or groups[group_index].responses != context.responses
                        or group_indices[group_index] != context.indices):
                    group_index = None
                if group_index is None:
                    group_index = len(groups)
                    groups.append(CandidateJudgeGroup(context.question, list(context.responses), [],
                        cache_scope=f"{stage_id}:{group_index}"))
                    segments.append([])
                    group_roles.append(set())
                    group_indices.append(list(context.indices))
                    slots[slot] = group_index
                group_roles[group_index].add(stage.role)
                groups[group_index].candidates.extend(record.candidate for record in records)
                segments[group_index].append((context, records))
        request_count = sum(len(group.responses) for group in groups if group.candidates)
        candidate_count = sum(len(group.candidates) for group in groups)
        unmerged_count = sum(len(context.responses) for rows in segments for context, records in rows if records)
        merged_groups = sum(sum(bool(records) for _, records in rows) > 1 for rows in segments)
        print(f"[RubricGenerator] Empirical Judge: {candidate_count} criteria across "
              f"{sum(map(len, segments))} proposals ({'+'.join(stage.role for stage in stages)}) in "
              f"{len(groups)} groups, {request_count} answer-level requests "
              f"({unmerged_count - request_count} saved by merging {merged_groups} groups).", flush=True)
        started = perf_counter()
        if candidate_count and not request_count:
            raise RuntimeError("Cannot judge new criteria without Solver responses")
        vectors = (self.evolver.judge_new_criterion_groups(groups)
                   if request_count else [[] for _ in groups])
        for rows, group_vectors in zip(segments, vectors, strict=True):
            if len(group_vectors) != sum(len(records) for _, records in rows):
                raise RuntimeError("Merged Judge column count mismatch")
            offset = 0
            for context, records in rows:
                for record, vector in zip(records, group_vectors[offset:offset + len(records)], strict=True):
                    _, empirical = self.evolver.verify_candidate_on_group(prompt_text=context.question,
                        candidate=record.candidate, source=context.operations[record.target_id],
                        rollout_responses=context.responses, expected_response_count=len(context.indices),
                        precomputed_scores=[{'criterion_0_score': score, 'judge_error': 0.0} for score in vector])
                    if empirical.get('score_vector') != vector:
                        raise RuntimeError("Empirical gate failed to preserve batched Judge scores")
                    record.empirical = deepcopy(empirical)
                offset += len(records)
        print(f"[RubricGenerator] Empirical Judge completed in {perf_counter()-started:.2f}s.", flush=True)
        metrics = {'rg/empirical_requests': float(request_count), 'rg/empirical_criteria': float(candidate_count),
                   'rg/empirical_requests_saved': float(unmerged_count - request_count),
                   'rg/empirical_merged_groups': float(merged_groups)}
        for stage in stages:
            metrics[f'rg/{stage.role}/empirical_criteria'] = float(sum(record.valid
                for proposal in stage.assessments for record in proposal.candidates))
        return metrics

    def _run(self, contexts: list[RGContext], *, role: str, timing_raw: dict, step: int) -> dict[str, float]:
        return self._run_roles({role: contexts}, timing_raw=timing_raw, step=step)

    def _run_roles(self, role_contexts: dict[str, list[RGContext]], *, timing_raw: dict,
                   step: int) -> dict[str, float]:
        """Propose/audit per role, judge all roles together, then score and commit.

        No rubric is committed (and no rollout is buffered) until every role has
        rewards. Commits follow role order: GAP roots first, then vertical children.
        """
        metrics, stages, dropped_groups = {}, [], 0
        for role, contexts in role_contexts.items():
            metrics.update({f"rg/{role}/groups": float(len(contexts)), f"rg/{role}/rollouts": 0.0,
                            ACCEPTED_KEYS[role]: 0.0})
            if contexts:
                stage = self._propose_and_audit(contexts, role=role, timing_raw=timing_raw, metrics=metrics)
                dropped = exclude_unavailable_groups(stage, self.diagnostics, step=step)
                metrics[f"rg/{role}/audit_unavailable_groups"] = float(dropped)
                metrics[f"rg/{role}/trainable_groups"] = float(len(stage.contexts))
                dropped_groups += dropped
                if stage.contexts:
                    stages.append(stage)
        total_groups = sum(len(contexts) for contexts in role_contexts.values())
        limit = float(getattr(self.config, "rubric_generator_max_unavailable_group_fraction", 0.1))
        # Request-level limits in the verifier miss the case where a few failed
        # batches each poison many groups; bound the dropped training signal too.
        if total_groups >= 8 and dropped_groups / total_groups > limit:
            path = self.write_diagnostics(step)
            raise RuntimeError(f"Meta audit unavailable for {dropped_groups}/{total_groups} RG groups at step {step} "
                               f"(> {limit:.0%}); diagnostics: {path}; meta failure logs under the RG log dir.")
        if not stages:
            return metrics
        started = perf_counter()
        metrics.update(self._judge_stages(stages))
        for stage in stages:
            for index, proposal in enumerate(stage.assessments):
                stage.results.append(score_rg_proposal(proposal,
                    output_tokens=int(stage.rollouts.data.batch["response_mask"][index].sum().item()),
                    token_budget=self.config.rubric_generator_soft_token_budget,
                    invalid_reward=self.config.rubric_generator_invalid_reward))
        timing_raw["rg_empirical_verify"] = perf_counter() - started
        for stage in stages:
            stage.rollouts.assign_rewards([result["reward"] for result in stage.results], role=stage.role)
        self.rollouts.extend(stage.rollouts for stage in stages)
        for stage in stages:
            self._commit_stage(stage, step=step, metrics=metrics)
        return metrics

    def _propose_and_audit(self, contexts: list[RGContext], *, role: str, timing_raw: dict,
                           metrics: dict[str, float]) -> RGStage:
        template = self.config.rubric_gap_proposal_template if role == "gap" else self.config.rubric_evolution_batch_proposal_template
        rollouts = self.generate([context.entry for context in contexts], template_path=template,
            max_tokens=self.config.rubric_evolution_proposer_max_tokens, timing_raw=timing_raw,
            timer_name=f"rg_{role}_rollout", temperature=self.config.rubric_generator_temperature,
            n=self.config.rubric_generator_rollout_n, keep_rollouts=True)
        n = rollouts.n
        if len(rollouts.texts) != len(contexts) * n:
            raise RuntimeError("RG rollout count mismatch.")
        assessments, plans = [], []
        for index, text in enumerate(rollouts.texts):
            context = contexts[index // n]
            proposal = parse_rg_proposal(text, role=role, operations=context.operations, budgets=context.budgets,
                weights=context.weights, existing_rubric=context.existing_rubric)
            assessments.append(proposal)
            plans.append(self._prepare_proposal_audit(context, proposal, role=role))
        operation_type = GAP_EXPAND if role == "gap" else CHILD_EVOLUTION
        request_count = sum(len(entries) for entries, _ in plans)
        print(f"[RubricGenerator] {role}: auditing {request_count} requests from "
              f"{len(assessments)} proposals using the configured verifier pool.", flush=True)
        started = perf_counter()
        audited = audit_proposal_batches([entries for entries, _ in plans], audit=self.audit,
            template_path=self.meta_router.template_for(operation_type),
            max_tokens=self.config.rubric_evolution_verifier_max_tokens)
        elapsed = perf_counter() - started
        timing_raw[f"rg_{role}_meta_verify"] = elapsed
        metrics[f"rg/{role}/meta_requests"] = float(request_count)
        print(f"[RubricGenerator] {role}: meta audit completed in {elapsed:.2f}s.", flush=True)
        # Apply in rollout order to avoid concurrent writes to the evolver.
        for index, (proposal, (_, chunks), texts) in enumerate(zip(assessments, plans, audited, strict=True)):
            context = contexts[index // n]
            self._apply_proposal_audit(context, chunks, texts, role=role, proposal=proposal)
        return RGStage(role, contexts, rollouts, assessments, n)

    def _commit_stage(self, stage: RGStage, *, step: int, metrics: dict[str, float]) -> None:
        role, rollouts, assessments, results, n = stage.role, stage.rollouts, stage.assessments, stage.results, stage.n
        rewards = [result["reward"] for result in results]
        for context_index, context in enumerate(stage.contexts):
            candidates = range(context_index * n, (context_index + 1) * n)
            # A parse failure should never be committed, even if penalties make
            # it numerically larger than an over-budget valid completion.
            eligible = [index for index in candidates if assessments[index].format_valid]
            winner = max(eligible, key=lambda index: (rewards[index], -index)) if eligible else None
            if winner is not None:
                metrics[ACCEPTED_KEYS[role]] += self._commit(context, assessments[winner], role=role, step=step)
            for index in candidates:
                proposal = assessments[index]
                self.diagnostics.append({"step": step, "role": role, "prompt_key": context.prompt_key,
                    "group_id": rollouts.group_ids[index], "sample_index": index % n,
                    "raw_completion": rollouts.texts[index],
                    "advantage": float(rollouts.data.batch["advantages"][index].sum().item()
                        / max(1, int(rollouts.data.batch["response_mask"][index].sum().item()))),
                    "selected": index == winner, **results[index], "format_error": proposal.error,
                    "target_ids": list(proposal.target_ids), "raw_counts": proposal.raw_counts,
                    "candidates": [{"target_id": record.target_id, "slot": record.slot,
                        "valid": record.valid, "accepted": record.accepted, "rejection": record.rejection,
                        "criterion": record.candidate, "pass_rate": record.empirical.get("pass_rate"),
                        "semantic": record.semantic, "empirical": record.empirical}
                        for record in proposal.candidates]})
        metrics[f"rg/{role}/rollouts"] = float(len(rollouts.data))
        for key in results[0]:
            metrics[f"rg/{role}/{key}_mean"] = float(np.mean([result[key] for result in results]))

    def _commit(self, context: RGContext, proposal: Any, *, role: str, step: int) -> int:
        if role == "gap":
            return sum(self.evolver.add_active_gap_root(prompt_key=context.prompt_key, prompt_text=context.question,
                candidate=record.candidate, global_step=step, semantic_verifier_info=record.semantic,
                empirical_verify_info=record.empirical, group_indices=context.indices)
                for record in proposal.candidates if record.accepted)
        payloads, meta_payloads, empirical_payloads = {}, {}, {}
        for target_id, target in context.targets.items():
            records = [record for record in proposal.candidates if record.target_id == target_id and record.accepted]
            output_key = "subcriteria" if target.state == CRITERION_STATE_DEAD else "refinements"
            key = (target.prompt_key, target.parent_index)
            payloads[key] = json.dumps({output_key: [record.candidate for record in records]}, ensure_ascii=False)
            for index, record in enumerate(records):
                meta_payloads[(*key, index)] = record.meta_payload
                empirical_payloads[(*key, index)] = deepcopy(record.empirical)
        metrics = self.evolver.evolve_targets(list(context.targets.values()), global_step=step,
            contrast_responses={}, proposal_payloads=payloads, proposal_verifier_payloads=meta_payloads,
            empirical_verifier_payloads=empirical_payloads, activate_immediately=True)
        return int(metrics.get("rubric_evolution/accepted_candidates", 0))
