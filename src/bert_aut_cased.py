import pandas as pd
from bert_score import score

#load csv
df = pd.read_csv("esnli_generated_explanation_deepseek.xlsx - EN.csv")

gold = df["Explanation_1"].astype(str).tolist()
generated = df["Generated_Explanation_EN"].astype(str).tolist()

# Default Bert model for english
P, R, F1 = score(
    generated, 
    gold, 
    lang="en", 
    model_type="bert-base-cased",
    num_layers=12
)

#bert score
df["BERT_Precision"] = P
df["BERT_Recall"] = R
df["BERT_F1"] = F1

#result
df.to_csv("bertscore_ceased.csv", index=False)

print("Average BERTScore F1:", df["BERT_F1"].mean())

