"""Full exact-match span and category+polarity evaluation on the held-out processed test split."""
import argparse,json
from pathlib import Path
import numpy as np
from .data import load_jsonl
from .decode import decode_bio
from .evaluate_aspects import score
from .encoding import encode_words,word_predictions

def main(args):
    import torch
    from transformers import AutoTokenizer,AutoModelForTokenClassification
    rows=load_jsonl(Path(args.processed)/f'{args.split}.jsonl')
    tok=AutoTokenizer.from_pretrained(args.checkpoint,use_fast=False)
    model=AutoModelForTokenClassification.from_pretrained(args.checkpoint)
    dev='cuda' if torch.cuda.is_available() else 'cpu';model.to(dev).eval()
    predicted=[];truncated=0
    for r in rows:
        e=encode_words(tok,r['words'],max_length=args.max_length)
        with torch.inference_mode():
            logits=model(input_ids=torch.tensor([e['input_ids']],device=dev),attention_mask=torch.tensor([e['attention_mask']],device=dev)).logits[0].cpu().numpy()
        guesses=word_predictions(logits,e['word_ids']);k=len(guesses)
        if k<len(r['words']):truncated+=1
        tags=[model.config.id2label[i] for _,i,_ in guesses]
        spans=decode_bio(r['words'][:k],r['word_offsets'][:k],tags)
        predicted.append({'text':r['text'],'spans':spans})
    result={'span':score(rows,predicted),'pair':score(rows,predicted,pair=True),'reviews':len(rows),'truncated_reviews':truncated}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf8') as f:
        for rec in predicted:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
    output.with_suffix('.metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--processed',default='data/processed');p.add_argument('--checkpoint',default='outputs/phobert/aspect/best')
    p.add_argument('--split',choices=['train','dev','test'],default='test');p.add_argument('--output',default='outputs/aspect_test_predictions.jsonl')
    p.add_argument('--max-length',type=int,default=256);main(p.parse_args())
