import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score


class ImbalancedAgreementCalculator:
    """
    Computes robust inter-rater agreement metrics for imbalanced binary data.
    
    Metrics included:
        - Observed Agreement (Pa)
        - Cohen's Kappa
        - PABAK
        - Gwet's AC1
        - Positive Specific Agreement
        - Negative Specific Agreement
    """

    def __init__(self, df: pd.DataFrame, positive_label=1):
        self.df = df
        self.positive_label = positive_label

    def _compute_confusion(self, y1, y2):
        tn, fp, fn, tp = confusion_matrix(
            y1, y2, labels=[0, self.positive_label]
        ).ravel()
        return tn, fp, fn, tp

    def compute_metrics(self, rater1_col: str, rater2_col: str):
        y1 = self.df[rater1_col]
        y2 = self.df[rater2_col]

        tn, fp, fn, tp = self._compute_confusion(y1, y2)
        total = tn + fp + fn + tp

        if total == 0:
            raise ValueError("No data available for agreement calculation.")

        # Observed Agreement
        pa = (tp + tn) / total

        # Cohen's Kappa
        kappa = cohen_kappa_score(y1, y2)

        # PABAK
        pabak = 2 * pa - 1

        # Gwet's AC1
        prop_yes_r1 = (tp + fp) / total
        prop_yes_r2 = (tp + fn) / total
        pi = (prop_yes_r1 + prop_yes_r2) / 2
        pe_gamma = 2 * pi * (1 - pi)

        if pe_gamma == 1:
            gwet_ac1 = 1.0
        else:
            gwet_ac1 = (pa - pe_gamma) / (1 - pe_gamma)

        # Specific Agreements
        pos_denom = (2 * tp + fp + fn)
        neg_denom = (2 * tn + fp + fn)

        pos_agree = (2 * tp / pos_denom) if pos_denom != 0 else 0
        neg_agree = (2 * tn / neg_denom) if neg_denom != 0 else 0

        return {
            "n": total,
            "observed_agreement": pa,
            "cohen_kappa": kappa,
            "pabak": pabak,
            "gwet_ac1": gwet_ac1,
            "positive_agreement": pos_agree,
            "negative_agreement": neg_agree,
            "confusion_matrix": {
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp
            }
        }

    def compute_multiple_dimensions(self, dimensions: list, suffix_1="_1"):
        results = {}

        for dim in dimensions:
            col1 = f"{dim}{suffix_1}"
            col2 = dim

            results[dim] = self.compute_metrics(col1, col2)

        return results