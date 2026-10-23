"""Two PhoBERT models; select best checkpoint on dev, evaluate final on test."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from .data import load_splits,load_jsonl,load_stars_csv
from .overall import LABELS,build_overall
from .encoding import encode_words
from .audit import audit

class Records:
    def __init__(self,items):self.items=items
    def __len__(self):return len(self.items)
    def __getitem__(self,i):return self.items[i]

def main(args):
    import torch
    from transformers import (AutoTokenizer,AutoModelForSequenceClassification,AutoModelForTokenClassification,
                              Trainer,TrainingArguments,DataCollatorForTokenClassification,DataCollatorWithPadding)
    from .segmentation import Segmenter, split_token_boundaries
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    raw=load_splits(args.data)
    leakage=audit(raw)
    if leakage['cross_split_duplicate_text_count'] and not args.allow_overlap:
        raise ValueError('Cross-split duplicates detected: inspect python -m src.audit; do not ignore without justification')
    output=Path(args.output)/args.task;output.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(args.model,use_fast=False)
    if args.task=='overall':
        stars=load_stars_csv(args.stars_csv,args.text_col,args.star_col) if args.stars_csv else []
        splits,provenance=build_overall(raw,stars,args.label_strategy)
        if min(map(len,splits.values()))==0:raise ValueError('Empty labeled split: check label strategy')
        seg=Segmenter(args.vncorenlp)
        datasets={}
        for k,rows in splits.items():
            texts = [
                ' '.join(
                    w for w, _, _ in split_token_boundaries(
            seg.aligned_words(r['text']), r['text']
        )
    )
    for r in rows
]
            enc=tokenizer(texts,truncation=True,max_length=args.max_length)
            datasets[k]=Records([{**{field:vals[i] for field,vals in enc.items()},'labels':LABELS.index(rows[i]['overall'])} for i in range(len(rows))])
        model=AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=len(LABELS),id2label=dict(enumerate(LABELS)),label2id={v:i for i,v in enumerate(LABELS)})
        collator=DataCollatorWithPadding(tokenizer)
        def compute(p):
            y=p.label_ids;pred=np.argmax(p.predictions,axis=-1)
            return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=range(len(LABELS)),average='macro',zero_division=0))}
        metric='macro_f1'
    else:
        folder=Path(args.processed)
        rows={k:load_jsonl(folder/f'{k}.jsonl') for k in ('train','dev','test')}
        names=json.loads((folder/'tags.json').read_text(encoding='utf8')); tag2id={v:i for i,v in enumerate(names)}
        datasets={}
        for k,samples in rows.items():
            records=[]
            for r in samples:
                e=encode_words(tokenizer,r['words'],r['tags'],args.max_length,tag2id)
                records.append({key:e[key] for key in ('input_ids','attention_mask','labels')})
            datasets[k]=Records(records)
        model=AutoModelForTokenClassification.from_pretrained(args.model,num_labels=len(names),id2label=dict(enumerate(names)),label2id=tag2id)
        collator=DataCollatorForTokenClassification(tokenizer)
        def compute(p):
            predicted=np.argmax(p.predictions,axis=-1);yy=[];pp=[]
            for gold,guess in zip(p.label_ids,predicted):
                mask=gold!=-100;yy.extend(gold[mask]);pp.extend(guess[mask])
            return {'token_accuracy':float(accuracy_score(yy,pp)), 'token_macro_f1':float(f1_score(yy,pp,labels=range(len(names)),average='macro',zero_division=0))}
        metric='token_macro_f1';provenance={'note':'Aspect gold only; exact-span and pair F1 use src.evaluate_model after training'}
    if any(len(datasets[k])==0 for k in datasets):raise ValueError('Empty split')
    cfg=TrainingArguments(output_dir=str(output/'checkpoints'),learning_rate=args.lr,per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,gradient_accumulation_steps=args.grad_accum,num_train_epochs=args.epochs,
        weight_decay=0.01,eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,
        metric_for_best_model=metric,greater_is_better=True,fp16=args.fp16 and torch.cuda.is_available(),
        report_to='none',save_total_limit=2,seed=args.seed,data_seed=args.seed)
    trainer=Trainer(model=model,args=cfg,train_dataset=datasets['train'],eval_dataset=datasets['dev'],data_collator=collator,compute_metrics=compute)
    t=time.time();trainer.train();elapsed=time.time()-t
    dev=trainer.evaluate(datasets['dev']);test=trainer.evaluate(datasets['test'],metric_key_prefix='test')
    best=output/'best';trainer.save_model(str(best));tokenizer.save_pretrained(str(best))
    info={'dev':dev,'test':test,'seconds_train':elapsed,'label_provenance':provenance,'config':vars(args),'overlap_audit':leakage}
    (output/'metrics.json').write_text(json.dumps(info,ensure_ascii=False,indent=2,default=float),encoding='utf8')
    print(json.dumps({'dev':dev,'test':test,'train_seconds':elapsed},indent=2,default=float))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['overall','aspect'],required=True)
    p.add_argument('--model',default='vinai/phobert-base-v2');p.add_argument('--data',default='data/visd4sa')
    p.add_argument('--processed',default='data/processed');p.add_argument('--vncorenlp',default='models/vncorenlp');p.add_argument('--stars-csv')
    p.add_argument('--text-col',default='comment');p.add_argument('--star-col',default='n_star');p.add_argument('--label-strategy',choices=['rating_then_majority','majority_only','rating_only'],default='rating_then_majority')
    p.add_argument('--output',default='outputs/phobert');p.add_argument('--max-length',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--grad-accum',type=int,default=2)
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--fp16',action='store_true');p.add_argument('--allow-overlap',action='store_true');main(p.parse_args())
