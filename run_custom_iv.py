#!/usr/bin/env python3
"""
Run GPT-J-6B with Iterative Vectors on custom sentences
"""

import torch
import pandas as pd
from pathlib import Path
from util import load_tokenizer_and_model  # from iterative-vectors repo

# ------------------ CONFIG ------------------
model_name = "gpt-j-6b"  # or "./models/gpt-j-6b-local" if stored locally
gpu_id = 0
output_csv = "custom_predictions.csv"

# Custom sentences you want to evaluate
custom_sentences = [
    "Wall Street stocks rise on strong earnings",
    "Lionel Messi scores winning goal for PSG",
    "NASA launches new satellite to study Mars",
    "Global leaders meet to discuss climate change"
]

# Wrap sentences into dataset format the repo expects
dataset = [{"text": s} for s in custom_sentences]

# ------------------ LOAD MODEL ------------------
class Args:
    model = model_name
    gpu = [gpu_id]
    no_half = False
    device_map = None
    flash_attention = False
    mem_limit = None
    no_split = []

args = Args()
tokenizer, model = load_tokenizer_and_model(args)

# ------------------ RUN IV ------------------
predictions = []
for item in dataset:
    text = item["text"]
    
    # Encode and generate
    inputs = tokenizer(text, return_tensors="pt").to(args.gpu[0])
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10)
    pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Save prediction
    predictions.append(pred_text)
    print(f"Input: {text}")
    print(f"Predicted: {pred_text}")
    print("----")

# ------------------ SAVE RESULTS ------------------
df = pd.DataFrame({
    "text": [item["text"] for item in dataset],
    "predicted_text": predictions
})
df.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")
