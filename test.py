import spacy
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModel

from chunkers import Chunker

chunker = Chunker()
ex = "Some dogs are running on a deserted beach."
print(ex)

# doc = chunker.nlp(ex)
# for token in doc:
#     print(token.text, token.tag_)

phrases = chunker.chunk(ex)
phrases_text = [phrase.text for phrase in phrases]
c_offsets = [(phrase.start_char, phrase.end_char) for phrase in phrases]

nlp = spacy.load("en_core_web_sm")
doc = nlp(ex)

# model = SentenceTransformer('all-MiniLM-L6-v2')
# embed = model.encode(ex, output_value='token_embeddings')
# print(embed)

# tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
# encoded_input = tokenizer(phrases_text, padding=True, truncation=True, max_length=256, return_tensors='pt')
#
# encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
# encoder2 = SentenceTransformer('all-mpnet-base-v2')
#
# with torch.no_grad():
#     model_output = encoder(**encoded_input)
#
# embeddings = encoder2.encode(phrases_text)
