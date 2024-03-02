import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    PrecisionRecallDisplay,
    precision_score,
    recall_score,
)

if TYPE_CHECKING:
    from typing import Tuple, List, Iterable

logger = logging.getLogger(__name__)


class MetricsBag:
    """
    This class provides all relevant methods to calculate the profit-metric, precision, recall, f1-scores
    and is able to visualize these metrics. It also integrates the ThresholdCalculatorForMinimalFalseNegatives.
    """

    def __init__(self, y: 'Iterable', y_hat: 'Iterable') -> None:
        """
        Calls the actual calculation-methods during initialization of the object.

        Parameters
        ----------
        y_truth : array_like
            Vector with ground truth
        y_hat : array_like
            Vector with predicted probabilities for positive class
        """

        self.y = y
        self.y_hat = y_hat

        self.thresholds = np.linspace(0, 1, num=30)

        self.f1_scores = self._calculate_f1_score()
        self.index_of_maximum_f1_score = np.argmax(self.f1_scores)

    def get_discrete_predictions_with_best_threshold(self) -> 'List[int]':
        """
        Creates discrete predictions from the probabilities (with the threshold that gives the highest profit).

        Returns
        -------
        list
            Discrete predictions according to the ideal threshold
        """
        return [1 if x >= self.profit_area.get_best_threshold() else 0 for x in self.y_hat]

    def evaluate(self) -> 'Tuple[plt.Figure,plt.axis]':
        """
        Shows all relevant metrics and numbers in a single plot. It is following 3 different ideas that correspondent
        with each row:
        1st row: Profit / Profit-Area
        2nd row: Get a maximum of 1% false negatives

        Returns
        -------
        tuple
            Tuple with matplotlib (Figure, Axes)
        """
        fig = plt.figure(figsize=(20, 4))
        gs = fig.add_gridspec(nrows=1, ncols=4, hspace=0.4)

        ax8 = fig.add_subplot(gs[0, 0])
        self._plot_f1_and_pr_curve(ax8)

        ax9 = fig.add_subplot(gs[0, 1])
        self._plot_f1_key_numbers(ax9)

        ax10 = fig.add_subplot(gs[0, 2])
        ax11 = fig.add_subplot(gs[0, 3])
        self._plot_f1_conf_mat(ax10, ax11)

        return fig, gs

    def _calculate_f1_score(self) -> 'np.ndarray':
        """
        Calculates F1-scores for different thresholds in the thresholds-array.
        """

        def calc_func(e: float):
            predictions = [1 if x > e else 0 for x in self.y_hat]
            return f1_score(self.y, predictions)

        results = np.array([calc_func(thresh) for thresh in self.thresholds])

        return results

    def _plot_f1_and_pr_curve(self, ax: plt.Axes) -> None:
        """
        Plots the F1-score over all thresholds.
        """
        ax.set_title("Precision-Recall-Curve and F1-Score")
        PrecisionRecallDisplay.from_predictions(self.y, self.y_hat, ax=ax)
        ax.plot(self.thresholds, self.f1_scores)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.legend(["Precision-Recall-Curve", "F1-Score over all thresholds"])

    def _plot_f1_key_numbers(self, ax: plt.Axes) -> None:
        """
        Shows key indicators (max. F1-score, Precision, Recall, the best threshold and the number of predictions)
        """
        max_precision, max_recall = self.get_max_precision_recall()
        text = f"Max. F1-score: {round(self.f1_scores[self.index_of_maximum_f1_score], 2)}\n" \
               f"Precision: {round(max_precision, 2)}\nRecall: {round(max_recall, 2)}" \
               f"\n\nBest threshold: {round(self.thresholds[self.index_of_maximum_f1_score], 2)}"
        ax.text(0.1, 0.3, text, size='xx-large')
        ax.set_title('Precision/Recall combination for max. F1-Score')
        ax.set_xticks([])
        ax.set_yticks([])

    def get_max_precision_recall(self) -> 'Tuple[float,float]':
        """
        Fetches the precision and recall values that lead to the highest F1-score.
        """
        predictions = [1 if x > self.thresholds[self.index_of_maximum_f1_score] else 0 for x in self.y_hat]
        return precision_score(self.y, predictions), recall_score(self.y, predictions)

    def _plot_f1_conf_mat(self, ax1: plt.Axes, ax2: plt.Axes) -> None:
        """
        Draws 2 confusion matrices with the predictions that result in the highest F1-score. One confusion matrix shows
        the raw numbers the other one is normalized over rows (=all positive labeled datapoints add up to 1 and all
        negative labeled datapoints add up to 1).
        """
        best_thresh_f1 = self.thresholds[self.index_of_maximum_f1_score]
        predictions = [1 if x > best_thresh_f1 else 0 for x in self.y_hat]

        ConfusionMatrixDisplay.from_predictions(self.y, predictions, ax=ax1)
        ax1.set_title('Confusion Matrix (for F1-score-threshold)')

        ConfusionMatrixDisplay.from_predictions(self.y, predictions, normalize='true', ax=ax2)
        ax2.set_title('Normalized CM (for F1-score-threshold)')
