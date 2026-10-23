"""Convert predicted BIO word tags back into raw character span coordinates."""

def decode_bio(words, offsets, tags, confidences=None):
    if not (len(words)==len(offsets)==len(tags)):
        raise ValueError('Words/offsets/tags lengths differ')
    if confidences is not None and len(confidences)!=len(tags):
        raise ValueError('Confidence length differs from tags')
    spans = []
    active = None
    scores = []

    def flush():
        nonlocal active, scores
        if active is not None:
            if scores:active['confidence'] = sum(scores)/len(scores)
            spans.append(active)
        active, scores = None, []

    for i, (tag, bounds) in enumerate(zip(tags, offsets)):
        start, end = map(int,bounds)
        if tag == 'O':
            flush(); continue
        if not (tag.startswith('B-') or tag.startswith('I-')) or '#' not in tag:
            flush();continue
        prefix, remainder = tag.split('-',1)
        category, polarity = remainder.rsplit('#',1)
        if active and prefix=='I' and active['category']==category and active['polarity']==polarity.lower():
            active['end'] = end
        else:
            flush()
            active = {'start':start,'end':end,'category':category,'polarity':polarity.lower()}
        if confidences is not None:scores.append(float(confidences[i]))
    flush()
    return spans
