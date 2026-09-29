"""Fixed first-answer-token color probabilities; not full-answer likelihoods."""
import torch


def score_colors(logits, tokenizer, colors=("red", "blue")):
    if not torch.isfinite(logits).all():
        raise RuntimeError('Non-finite color-score logits')
    candidates = {}
    if len(set(colors)) != len(colors) or len(colors) < 2:
        raise ValueError('Expected distinct score candidates')
    for color in colors:
        ids = set()
        for spelling in (color, color.capitalize(), ' '+color, ' '+color.capitalize()):
            encoded = tokenizer.encode(spelling, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f'Color score spelling is not a single token: {spelling!r}')
            ids.add(encoded[0])
        candidates[color] = sorted(ids)
    if sum(map(len, candidates.values())) != len({i for ids in candidates.values() for i in ids}):
        raise ValueError('Color candidate token sets overlap')
    logp = logits.log_softmax(-1)
    masses = {c:float(torch.logsumexp(logp[ids],0)) for c,ids in candidates.items()}
    return {'method':'first_answer_token_variant_mass_v1', 'token_ids':candidates,
            'log_probability':masses, 'probability':{c:float(torch.exp(torch.tensor(v))) for c,v in masses.items()}}


def compare_scores(scores, clean, original, donor):
    margin=scores['log_probability'][donor]-scores['log_probability'][original]
    base=clean['log_probability'][donor]-clean['log_probability'][original]
    return {**scores, 'original_color':original, 'donor_color':donor,
            'donor_minus_original_log_odds':margin, 'clean_log_odds':base,
            'change_from_clean':margin-base}
