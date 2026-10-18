"""Build serialized word-tag sequences for token classification.
Raw review text and raw offsets are preserved for later highlighting.
"""
import argparse
import json
from pathlib import Path
from collections import Counter
from .data import load_splits
from .segmentation import Segmenter, split_token_boundaries, word_labels


def main(args):
    out = Path(args.output).resolve()
    splits = load_splits(args.data)
    seg = Segmenter(args.vncorenlp)
    out.mkdir(parents=True, exist_ok=True)
    tags = {'O'}
    counts = Counter()
    for name,rows in splits.items():
        with (out/f'{name}.jsonl').open('w',encoding='utf8') as f:
            for row in rows:
                try:
                    words = seg.aligned_words(row['text'])
                    words = split_token_boundaries(words, row['text'])
                    labels = word_labels(words, row['spans'], row['text'])
                    
                    record={**row,'words':[w for w,_,_ in words], 'word_offsets':[[a,b] for _,a,b in words], 'tags':labels}
                    tags.update(labels)
                    f.write(json.dumps(record,ensure_ascii=False)+'\n')
                    counts[f'{name}_accepted']+=1
                except ValueError as exc:
                    counts[f'{name}_rejected']+=1
                    counts[f'{name}_reason_{str(exc).split(":")[0]}'] += 1
                    if not args.allow_rejected: raise ValueError(f'{name} #{counts[f"{name}_accepted"]+counts[f"{name}_rejected"]}: {exc}') from exc
        
    (out/'tags.json').write_text(json.dumps(['O']+sorted(tags-{'O'}),ensure_ascii=False,indent=2),encoding='utf8')
    (out/'preprocessing_report.json').write_text(json.dumps(dict(counts),indent=2),encoding='utf8')
    print(dict(counts))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/visd4sa');p.add_argument('--vncorenlp',default='models/vncorenlp')
    p.add_argument('--output',default='data/processed');p.add_argument('--allow-rejected',action='store_true')
    main(p.parse_args())
