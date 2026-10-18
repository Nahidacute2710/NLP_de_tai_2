"""Explicit weak-label provenance for document-level sentiment."""
from collections import Counter, defaultdict
from .data import norm, star_polarity

LABELS = ['negative', 'neutral', 'positive']


def majority_label(spans):
    counts = Counter(s['polarity'] for s in spans)
    if not counts:
        return None
    best = max(counts.values())
    winners = [k for k, v in counts.items() if v == best]
    return winners[0] if len(winners) == 1 else None


def build_overall(splits, star_entries=None, strategy='rating_then_majority'):
    """Return labeled rows and per-split label provenance; no silent tie resolution.

    Rating joins use exact NFC/casefold/space normalized text, accepting only one
    distinct star-derived polarity per normalized review. Labels from the same
    text are never trusted across contradictory star records.
    """
    if strategy not in ('rating_then_majority', 'majority_only', 'rating_only'):
        raise ValueError(f'Invalid strategy: {strategy}')
    ratings = defaultdict(set)
    for text, stars in star_entries or []:
        ratings[norm(text)].add(star_polarity(int(stars)))
    output, stats = {}, {}
    for split, rows in splits.items():
        labeled = []
        tally = Counter()
        for row in rows:
            key = norm(row['text'])
            choices = ratings.get(key, set())
            if strategy != 'majority_only' and len(choices) == 1:
                label, source = next(iter(choices)), 'rating_exact'
            elif strategy == 'rating_only':
                label, source = None, None
            elif len(choices) > 1:
                # Conflicting ratings: do not let a fallback conceal ambiguity.
                label, source = None, None
                tally['conflicting_ratings'] += 1
            else:
                label, source = majority_label(row['spans']), 'majority_aspects'
            if label is not None:
                labeled.append({**row, 'overall': label, 'overall_source': source})
                tally[source] += 1
                tally[f'label_{label}'] += 1
            else:
                tally['excluded_unlabeled_or_tied'] += 1
        output[split] = labeled
        tally['raw'] = len(rows)
        stats[split] = dict(tally)
    return output, stats
