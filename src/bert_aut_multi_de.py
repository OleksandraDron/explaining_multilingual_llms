import pandas as pd
from bert_score import score

#load csv
df = pd.read_excel("esnli_generated_explanation_deepseek_de.xlsx", sheet_name="DE")

gold = df["Explanation_1"].astype(str).tolist()
generated = df["Generated_Explanation_DE"].astype(str).tolist()

# Default Bert model for english
P, R, F1 = score(
    generated, 
    gold, 
    lang="None", 
    model_type="bert-base-multilingual-cased"
)

#bert score
df["BERT_Precision"] = P
df["BERT_Recall"] = R
df["BERT_F1"] = F1

#result
df.to_csv("bertscore_multi_de.csv", index=False)

print("Average BERTScore F1:", df["BERT_F1"].mean())

