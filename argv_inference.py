# Derived from: https://github.com/naver/splade/blob/eb74d1ab31c42b9b94df7c0d67cc582b2c972dda/inference_SPLADE.ipynb
# Dependencies
import sys
import json
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


# Model
# Extends torch module
# Overrides `forward()`
class Splade(torch.nn.Module):

    def __init__(self, model_type_or_dir, agg="max"):
        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)
        assert agg in ("sum", "max")
        self.agg = agg

    def forward(self, **kwargs):
        out = self.transformer(**kwargs)["logits"] # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        if self.agg == "max":
            values, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
            return values
            # 0 masking also works with max because all activations are positive
        else:
            return torch.sum(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)


# Model folder or repo
#
#model_type_or_dir = "naver/neuclir22-splade-ru"
model_type_or_dir = "naver/splade-cocondenser-ensembledistil"

# Model loading
#
model = Splade(model_type_or_dir)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

def inspect_json(sorted_d, mag):
  elem = {}
  for k, v in sorted_d.items():
    elem[reverse_voc[k]] = round(v, 2)

  _json = { 'elements': elem, 'magnitude': mag }

  print("JSON representation:")
  print(json.dumps(_json, indent=4, ensure_ascii=False))


# example document from MS MARCO passage collection (doc_id = 8003157)
# doc = "Glass and Thermal Stress. Thermal Stress is created when one area of a glass pane gets hotter than an adjacent area. If the stress is too great then the glass will crack. The stress level at which the glass will break is governed by several factors."

# Use argv instead
doc=sys.argv[1]


# Tokenize
tokenized_data = tokenizer(doc, return_tensors="pt")
size = tokenized_data.input_ids.size()[1]

print("Tokens in total: ", size)
# print(tokenized_data)


# Split lots of tokens into 512-token slices
# 512 tokens is the limit of this model
slices = []
offset = 0

while offset < size:
  tgt = min(offset + 512, size)
  slice = {}
  slice['input_ids'] = tokenized_data.input_ids[:,offset:tgt].clone()
  slice['attention_mask'] = tokenized_data.attention_mask[:,offset:tgt]

  # Scrutiny paid off:
  # We saw a big drop in dimension count with initial slicing
  # Turns out that we were missing token 102, aka [SEP], that should go at the end
  # Adding it back forcibly brings the dimensions back up. The model appears to have heavy dependence on it being there.
  # Additionally, we are missing token 101 from the start of any consecutive slices, but it doesn't seem to make much difference
  # Be sure to execute .clone() first in order to avoid edition the original tokenized_data, since pytorch uses "views" for slicing
  if tgt - offset == 512:
    slice['input_ids'][0,511] = 102

  offset = offset + 511
  slices.append(slice)


# Print tokenized input so you can see what the model sees
# ['[CLS]', 'word1', 'word2', 'word3', ..., '[SEP]']
def dump_slice(slice):
  token_ids = slice['input_ids'].tolist()[0]
  tokens = []
  for id in token_ids:
    tokens.append(reverse_voc[id])

  print(tokens)

for slice in slices:
  dump_slice(slice)


# Evaluate 512-token slices and add up the vectors each one produced
# Model returns sparse vectors of 30522 elements
def process_slice(data):
  with torch.no_grad():
      doc_rep = model(**data).squeeze()

  #print("Slice!")
  return doc_rep

z = torch.zeros(tokenizer.vocab_size)

for slice in slices:
  z = z + process_slice(slice)

# Precompute the magnitude
# 1. Dot product self
# 2. Take square root
# 3. Tensor -> float
mag = float(z.dot(z)**0.5)

# get the number of non-zero dimensions in the rep:
col = torch.nonzero(z).squeeze().cpu().tolist()
print("number of actual dimensions: ", len(col))

# now let's inspect the bow representation:
weights = z[col].cpu().tolist()
d = {k: v for k, v in zip(col, weights)}
sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}

inspect_json(sorted_d, mag)
