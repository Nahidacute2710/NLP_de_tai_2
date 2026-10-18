"""Review-level overlap audit: preserve official splits; never silently de-duplicate test."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from .data import load_splits, norm


def audit(splits):
    by_text = defaultdict(list)
    duplicates = {}
    for name, rows in splits.items():
        seen = set()
        dup = 0
        for i, r in enumerate(rows):
            k = norm(r['text'])
            dup += k in seen
            seen.add(k)
            by_text[k].append((name, i))
        duplicates[name] = dup
    crossing = {key: list(items) for key, items in by_text.items()
                if len({split for split, _ in items}) > 1}
    return {'sizes': {k:len(v) for k,v in splits.items()},
            'within_split_duplicate_counts': duplicates,
            'cross_split_duplicate_text_count': len(crossing),
            'cross_split_examples': list(crossing.values())[:30]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/visd4sa');p.add_argument('--out',default='outputs/audit.json');a=p.parse_args()
    result=audit(load_splits(a.data));o=Path(a.out);o.parent.mkdir(parents=True,exist_ok=True)
    o.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
