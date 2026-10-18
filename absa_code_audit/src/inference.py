"""Local inference contract for dashboard developer: predict_review(text). No hosted LLM APIs."""
from email.mime import text
from pathlib import Path
from .encoding import encode_words,word_predictions
from .decode import decode_bio
from .segmentation import split_token_boundaries

class ABSAInference:
    def __init__(self,model_root='outputs/phobert',vncorenlp='models/vncorenlp',device=None,max_length=256):
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification,AutoModelForTokenClassification
        from .segmentation import Segmenter
        root=Path(model_root);self.seg=Segmenter(vncorenlp);self.max_length=max_length
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.overall_path=root/'overall'/'best';self.aspect_path=root/'aspect'/'best'
        self.otok=AutoTokenizer.from_pretrained(str(self.overall_path),use_fast=False)
        self.atok=AutoTokenizer.from_pretrained(str(self.aspect_path),use_fast=False)
        self.overall=AutoModelForSequenceClassification.from_pretrained(str(self.overall_path)).to(self.device).eval()
        self.aspect=AutoModelForTokenClassification.from_pretrained(str(self.aspect_path)).to(self.device).eval()

    def predict_review(self,text):
        import torch
        if not isinstance(text,str) or not text.strip():raise ValueError('Input must be nonempty text')
        words = self.seg.aligned_words(text)
        words = split_token_boundaries(words, text)
        if not words:return {'text':text,'overall_sentiment':None,'overall_confidence':None,'aspects':[]}
        joined=' '.join(w for w,_,_ in words)
        oe=self.otok(joined,return_tensors='pt',truncation=True,max_length=self.max_length)
        oe={k:v.to(self.device) for k,v in oe.items()}
        with torch.inference_mode():
            probs=self.overall(**oe).logits.softmax(-1)[0]
        idx=int(probs.argmax()); overall=self.overall.config.id2label[idx]
        awords=[w for w,_,_ in words];offsets=[(a,b) for _,a,b in words]
        ae=encode_words(self.atok,awords,max_length=self.max_length)
        ids=ae['input_ids'];mask=ae['attention_mask']
        with torch.inference_mode():
            logits=self.aspect(input_ids=torch.tensor([ids],device=self.device),attention_mask=torch.tensor([mask],device=self.device)).logits[0].cpu().numpy()
        predicted=word_predictions(logits,ae['word_ids']); tags=['O']*len(words); conf=[0.0]*len(words)
        for w,tag_id,c in predicted:tags[w]=self.aspect.config.id2label[tag_id];conf[w]=c
        # Word segments beyond max_length are omitted (not silently predicted as O).
        predicted_count=len(predicted); spans=decode_bio(awords[:predicted_count],offsets[:predicted_count],tags[:predicted_count],conf[:predicted_count])
        for s in spans:s['text']=text[s['start']:s['end']]
        return {'text':text,'overall_sentiment':overall.upper(),'overall_confidence':round(float(probs[idx]),4),'aspects':spans,'truncated':predicted_count<len(words)}
