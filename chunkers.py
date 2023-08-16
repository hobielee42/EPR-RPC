import spacy
from numpy import arange


class Chunker:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)

    def chunk(self, sent: str):
        doc = self.nlp(sent)

        phrase_indices = []
        pps = []
        nps = []
        vps = []
        ops = []

        for np in doc.noun_chunks:
            if np.start > 0 and doc[np.start - 1].tag_ == 'IN':
                pps.append(doc[np.start - 1:np.end])
                phrase_indices += arange(np.start - 1, np.end).tolist()
            else:
                nps.append(np)
                phrase_indices += arange(np.start, np.end).tolist()

        for token in doc:
            if token.pos_ == 'VERB':
                start = token.i
                end = start + 1
                if start - 1 > 0 and doc[start - 1].dep_ == 'neg':
                    start -= 1
                    if start - 1 > 0 and doc[start - 1].pos_ == 'AUX':
                        start -= 1
                elif start - 1 > 0 and doc[start - 1].pos_ == 'AUX':
                    start -= 1
                if end + 1 <= len(doc) and doc[end].tag_ == 'RP':
                    end += 1

                vps.append(doc[start:end])
                phrase_indices += arange(start, end).tolist()

        open_class_tags = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB']
        for token in doc:
            if token.pos_ in open_class_tags and token.i not in phrase_indices:
                ops.append(doc[token.i:token.i + 1])
                phrase_indices.append(token.i)

        print('doc:', doc)
        print('pps:', pps)
        print('nps:', nps)
        print('vps:', vps)
        print('ops:', ops)
        print(phrase_indices)

        phrases = pps + nps + vps + ops
        phrases.sort(key=lambda p: p.start)
        print('phrases:', phrases)

        return phrases
