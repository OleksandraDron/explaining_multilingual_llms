import pandas as pd
from bert_score import BERTScorer


class BERTScoreEvaluator:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        reference_column: str,
        candidate_column: str,
        model_type: str = "roberta-large",
        lang: str = "en",
        rescale: bool = True,
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.reference_column = reference_column
        self.candidate_column = candidate_column
        self.model_type = model_type
        self.lang = lang
        self.rescale = rescale

        self.df = None
        self.scorer = None

    def load_data(self):
        self.df = pd.read_excel(self.input_file)
        self.df = self.df.dropna(
            subset=[self.reference_column, self.candidate_column]
        )

        print(f"Loaded {len(self.df)} valid pairs for BERTScore calculation.")

    def initialize_scorer(self):
        self.scorer = BERTScorer(
            model_type=self.model_type,
            lang=self.lang,
            rescale_with_baseline=self.rescale,
        )

    def compute_scores(self):
        references = self.df[self.reference_column].astype(str).tolist()
        candidates = self.df[self.candidate_column].astype(str).tolist()

        P, R, F1 = self.scorer.score(candidates, references)

        self.df["BERTScore_P"] = P.tolist()
        self.df["BERTScore_R"] = R.tolist()
        self.df["BERTScore_F1"] = F1.tolist()

        print(f"\nBERTScore Mean Precision: {P.mean():.4f}")
        print(f"BERTScore Mean Recall:    {R.mean():.4f}")
        print(f"BERTScore Mean F1:        {F1.mean():.4f}")

    def save_results(self):
        self.df.to_excel(self.output_file, index=False)
        print(f"\n✅ Results saved to: {self.output_file}")

    def run(self):
        self.load_data()
        self.initialize_scorer()
        self.compute_scores()
        self.save_results()

  def main():
    evaluator = BERTScoreEvaluator(
        input_file="data/English_dataset_extension.xlsx",
        output_file="data/BERTScore_roberta-large_rescaled.xlsx",
        reference_column="Explanation_1",
        candidate_column="Generated_Explanation_EN",
        model_type="roberta-large",
        lang="en",
        rescale=True,
    )

    evaluator.run()


if __name__ == "__main__":
    main()
