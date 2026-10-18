"""Exact span and aspect+polarity pair precision/recall/F1 on raw offsets.
Input predictions JSONL with text and spans, aligned line-by-line with gold.
"""
import argparse,json
from .data import read_aspects,load_jsonl

def score(gold,pred,pair=False):
    if len(gold) != len(pred):
        raise ValueError('Gold/prediction review counts do not match')
    tp=fp=fn=0
    for g,p in zip(gold,pred):
        def key(s):
            return (s['start'],s['end'],s['category'],s['polarity']) if pair else (s['start'],s['end'])
        a={key(s) for s in g['spans']};b={key(s) for s in p['spans']}
        tp+=len(a&b);fp+=len(b-a);fn+=len(a-b)
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    return {'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0,
            'tp':tp,'fp':fp,'fn':fn}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gold',required=True);p.add_argument('--predictions',required=True);a=p.parse_args()
    gold=read_aspects(a.gold);pred=load_jsonl(a.predictions)
    if len(gold)!=len(pred):raise ValueError('Gold/prediction review counts do not match')
    if any(g['text']!=r['text'] for g,r in zip(gold,pred)):raise ValueError('Text order mismatch')
    print(json.dumps({'aspect_span':score(gold,pred),'aspect_sentiment_pair':score(gold,pred,True)},indent=2))
