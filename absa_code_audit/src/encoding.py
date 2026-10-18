"""Shared PhoBERT encoding and word/subword label alignment.
Input is VnCoreNLP-segmented words, including underscore-joined multi-syllable words.
"""
def encode_words(tokenizer, words, labels=None, max_length=256, tag2id=None):
    if labels is not None and len(words) != len(labels):
        raise ValueError('Word/tag lengths differ')
    if max_length < 3:
        raise ValueError('max_length must be >= 3')
    ids=[tokenizer.bos_token_id]; word_ids=[None]; gold=[-100]
    for i,word in enumerate(words):
        # Prefix whitespace after first word to preserve RoBERTa BPE word boundaries.
        token_ids=tokenizer.encode((' ' if i else '')+word,add_special_tokens=False)
        if not token_ids:raise ValueError('Tokenizer returned no tokens for '+repr(word))
        ids.extend(token_ids);word_ids.extend([i]*len(token_ids))
        if labels is not None:
            gold.extend([tag2id[labels[i]]] + [-100]*(len(token_ids)-1))
        else:gold.extend([-100]*len(token_ids))
    ids.append(tokenizer.eos_token_id);word_ids.append(None);gold.append(-100)
    if len(ids)>max_length:
        # Never keep a partial word: the last subword could otherwise be cut.
        keep=max_length-1
        while keep>1 and word_ids[keep-1] is not None and word_ids[keep-1] == word_ids[keep]:
            keep-=1
        ids=ids[:keep]+[tokenizer.eos_token_id]
        word_ids=word_ids[:keep]+[None]
        gold=gold[:keep]+[-100]
    return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':gold,'word_ids':word_ids}


def word_predictions(logits, word_ids):
    """Select logits from first subword for each preserved word."""
    import numpy as np
    chosen=[];seen=set()
    for pos,w in enumerate(word_ids):
        if w is not None and w not in seen:
            chosen.append((w,logits[pos]));seen.add(w)
    return [(w,int(np.argmax(scores)),float(np.max(_softmax(scores)))) for w,scores in chosen]

def _softmax(scores):
    import numpy as np
    e=np.exp(scores-np.max(scores));return e/e.sum()
