import torch
import flag_gems
from transformers import AutoTokenizer, BertConfig, BertModel

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
config = BertConfig()
model = BertModel(config)
model.to("cuda")
model.eval()

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to("cuda")
with torch.no_grad():
    ref_outputs = model(**inputs)

print("=======================================")
flag_gems.enable()
with torch.no_grad():
    res_outputs = model(**inputs)

maxdiff = torch.max(
    torch.abs(ref_outputs.last_hidden_state - res_outputs.last_hidden_state)
)

succeed = True

if (
    torch.allclose(
        ref_outputs.last_hidden_state,
        res_outputs.last_hidden_state,
        atol=1e-2,
        rtol=1e-2,
    )
    is False
):
    score = torch.nn.functional.cosine_similarity(
        ref_outputs.last_hidden_state.flatten(),
        res_outputs.last_hidden_state.flatten(),
        dim=0,
        eps=1e-6,
    )
    succeed = score >= 0.99

if succeed:
    print("##### SUCCEED #####")
else:
    print("##### FAILED ######")
    print("Max diff:", maxdiff)
