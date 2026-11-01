"""Recover audit formatting failures without fabricating generator rewards."""
from copy import deepcopy


class UnavailableAudit(str):
    """Internal-only service-failure marker; never accepted from model JSON."""


def single_candidate_entry(entry: dict, candidate: dict) -> dict:
    """Rebuild a one-candidate audit with the same instructions as the batch.

    Peers are only the same-completion candidates listed BEFORE this one, so the
    candidate is never compared with itself and the batch rule "reject redundant
    later candidates" keeps the same meaning.
    """
    from .rubric_meta_verifier import RubricMetaVerifierRouter, format_peer_candidates_hint

    result = deepcopy(entry)
    context = result['meta_context']
    peers = context.get('peer_candidates', [])
    ids = [peer.get('candidate_id') for peer in peers]
    earlier = peers[:ids.index(candidate['candidate_id'])] if candidate['candidate_id'] in ids else list(peers)
    context['candidates'] = [deepcopy(candidate)]
    context['peer_candidates'] = deepcopy(earlier)
    result['hint'] = RubricMetaVerifierRouter(None, None).build_batch_hint(
        operation_type=context['operation_type'], existing_rubric=context['existing_rubric'],
        candidate_records=context['candidates'])
    if earlier:
        result['hint'] += '\n\n' + format_peer_candidates_hint(earlier)
    return result


def exclude_unavailable_groups(stage, diagnostics: list, *, step: int) -> int:
    """Drop entire prompt/role GRPO groups before judging or computing rewards.

    Every retained group keeps all n trajectories. Removed groups contribute
    neither policy nor KL loss, and no patch is committed from them.
    """
    bad = {i // stage.n for i, proposal in enumerate(stage.assessments) if proposal.audit_error}
    if not bad:
        return 0
    for group in sorted(bad):
        errors = [p.audit_error for p in stage.assessments[group*stage.n:(group+1)*stage.n] if p.audit_error]
        for i in range(group*stage.n, (group+1)*stage.n):
            diagnostics.append(dict(step=step, role=stage.role,
                prompt_key=stage.contexts[group].prompt_key, group_id=stage.rollouts.group_ids[i],
                sample_index=i % stage.n, raw_completion=stage.rollouts.texts[i],
                audit_status='unavailable_group', audit_errors=errors,
                selected=False, reward=None, advantage=None))
    keep_groups = [i for i in range(len(stage.contexts)) if i not in bad]
    keep_rows = [g*stage.n+k for g in keep_groups for k in range(stage.n)]
    stage.contexts = [stage.contexts[i] for i in keep_groups]
    stage.assessments = [stage.assessments[i] for i in keep_rows]
    if keep_rows:
        import torch
        stage.rollouts.data = stage.rollouts.data.index_select(torch.tensor(keep_rows, dtype=torch.long))
        stage.rollouts.texts = [stage.rollouts.texts[i] for i in keep_rows]
        stage.rollouts.group_ids = [stage.rollouts.group_ids[i] for i in keep_rows]
    return len(bad)
