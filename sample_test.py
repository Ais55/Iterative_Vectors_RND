import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from util import Verbalizer

# 1. Load model and tokenizer
model_name = "EleutherAI/gpt-j-6b"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.to(device)
model.eval()

# 2. Prepare sample
sample = [{"input": "Apple is looking at buying a startup in the UK.",
           "output": "Business"}]

# 3. Create verbalizer
verbalizer = Verbalizer(tokenizer)

# 4. Tokenize sample
full_token, ans_mask, io_mask = verbalizer(sample)

# Move tensors to device
full_token = {k: v.to(device) for k, v in full_token.items()}
ans_mask = ans_mask.to(device)
io_mask = io_mask.to(device)

# 5. Run model
with torch.no_grad():
    outputs = model(**full_token)
    logits = outputs.logits

# 6. Decode prediction
pred_ids = logits.argmax(dim=-1)[0]  # greedy decoding
pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)

print("Input Text:", sample[0]["input"])
print("Predicted Text:", pred_text)
print("Expected Output:", sample[0]["output"])
