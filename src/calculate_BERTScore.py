import pandas as pd
from bert_score import BERTScorer

# === file path ===
input_file = "data/English_dataset_extension.xlsx"
output_file = "data/BERTScore_roberta-large_rescaled.xlsx"

# === 1. read file ===
df = pd.read_excel(input_file)
df = df.dropna(subset=["Explanation_1", "Generated_Explanation_EN"])

references = df["Explanation_1"].astype(str).tolist()
candidates = df["Generated_Explanation_EN"].astype(str).tolist()

print(f"Loaded {len(df)} valid pairs for BERTScore calculation.")

# === 2. initilize BERTScorer ===
scorer = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True)

# === 3. calculate BERTScore ===
P, R, F1 = scorer.score(candidates, references)

# === 4. save to DataFrame ===
df["BERTScore_P"] = P.tolist()
df["BERTScore_R"] = R.tolist()
df["BERTScore_F1"] = F1.tolist()

# === 5. calculate means ===
print(f"\nBERTScore Mean Precision: {P.mean():.4f}")
print(f"BERTScore Mean Recall:    {R.mean():.4f}")
print(f"BERTScore Mean F1:        {F1.mean():.4f}")

# === 6. save result ===
df.to_excel(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")
