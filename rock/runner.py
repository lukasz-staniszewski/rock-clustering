import argparse
import os
import time
from collections import OrderedDict
from datetime import datetime
from typing import Dict

from datasets import (
    CONGRESSIONAL_DATASET,
    CSV_DATASET,
    MUSHROOM_DATASET,
    get_congressional_dataset,
    get_csv_dataset,
    get_mushroom_dataset,
)
from process_data import get_rock_input, get_rock_output

from rock import metrics
from rock.rock_algorithm import RockAlgorithm


class Runner:
    def __init__(
        self,
        dataset_name,
        theta,
        k,
        approx_fn_name,
        split_train,
        dataset_path=None,
        calculate_metrics=True,
        skip_outliers=False,
    ):
        self.dataset_name = dataset_name
        self.theta = theta
        self.k = k
        self.approx_fn_name = approx_fn_name
        self.split_train = split_train
        self.dataset_path = dataset_path
        self.calculate_metrics = calculate_metrics
        self.skip_outliers = skip_outliers
        self.run_name = self._get_run_name()

        self._config_data()
        self._config_fn()

        self.rock_algorithm = RockAlgorithm(dataset=self.data, theta=self.theta, k=self.k, approx_fn=self.approx_fn)

    def _config_data(self):
        if self.dataset_name == MUSHROOM_DATASET:
            self.dataset = get_mushroom_dataset()
        elif self.dataset_name == CONGRESSIONAL_DATASET:
            self.dataset = get_congressional_dataset()
        elif self.dataset_name == CSV_DATASET:
            if self.dataset_path is None:
                raise ValueError("Dataset path not found but 'csv' dataset was selected")
            self.dataset = get_csv_dataset(self.dataset_path)
        else:
            raise ValueError("Dataset not found")
        self.data = get_rock_input(dataset=self.dataset, split_train=self.split_train)

    def _config_fn(self):
        if self.approx_fn_name == "rational_add":
            self.approx_fn = metrics.RATIONAL_ADD
        elif self.approx_fn_name == "rational_sub":
            self.approx_fn = metrics.RATIONAL_SUB
        elif self.approx_fn_name == "rational_exp":
            self.approx_fn = metrics.RATIONAL_EXP
        elif self.approx_fn_name == "rational_sin":
            self.approx_fn = metrics.RATIONAL_SIN
        else:
            raise ValueError("Approximation function not found")

    def _save_results_to_csv(self) -> None:
        df_out = get_rock_output(dataset=self.dataset, rock_output=self.results)
        output_path = os.path.join(os.getcwd(), "results", f"{self.run_name}.csv")
        df_out.to_csv(output_path, index=False, header=True)

    def _save_metrics_to_csv(self, metrics_dict: Dict[str, float]) -> None:
        df_dict = OrderedDict(
            **{
                "run_name": self.run_name,
                "dataset_name": self.dataset_name,
                "theta": self.theta,
                "k": self.k,
                "approx_fn_name": self.approx_fn_name,
                "split_train": self.split_train,
            }
        )
        df_dict.update(metrics_dict)
        output_path = os.path.join(os.getcwd(), "metrics", f"{self.run_name}.csv")
        df_dict.to_csv(output_path, index=False, header=True)

    def _get_run_name(self):
        return (
            datetime.now().strftime("%Y%m%d%H%M")
            + f"_{self.dataset_name}_theta{self.theta}_k{self.k}_{self.approx_fn_name}_split{self.split_train}"
        )

    def _calculate_metrics(self, runtime_sec: float) -> None:
        no_target = self.data.data_train[0].y is None
        out_metrics = {}
        if no_target:
            print("No target found, using only internal metrics")
        else:
            print("Target found, using internal and external metrics")
            purity_val = metrics.purity(points=self.results, skip_outliers=self.skip_outliers)
            out_metrics[metrics.METRIC_PURITY] = purity_val
            print(f"Metrics | Purity: {purity_val:.3f}")

        silhouette_val = metrics.silhouette(points=self.results, skip_outliers=self.skip_outliers)
        out_metrics[metrics.METRIC_SILHOUETTE] = silhouette_val
        print(f"Metrics | Silhouette: {silhouette_val:.3f}")
        print(f"Metrics | Runtime: {runtime_sec:.3f} seconds")
        out_metrics["runtime_sec"] = f"{runtime_sec:.3f}"
        self._save_metrics_to_csv(metrics_dict=out_metrics)

    def run(self) -> None:
        if self.calculate_metrics:
            start_time = time.process_time()
        self.rock_algorithm.create_clusters()
        self.results = self.rock_algorithm.collect_results(skip_outliers=self.skip_outliers)
        if self.calculate_metrics:
            end_time = time.process_time()
        self._save_results_to_csv()
        if self.calculate_metrics:
            self._calculate_metrics(runtime_sec=end_time - start_time)


def main(args: argparse.Namespace):
    runner = Runner(
        dataset_name=args.dataset,
        theta=args.theta,
        k=args.k,
        approx_fn_name=args.approx_fn,
        split_train=args.split_train,
        dataset_path=args.dataset_path,
        calculate_metrics=args.calculate_metrics,
        skip_outliers=args.skip_outliers,
    )
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Rock algorithm")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to run the algorithm",
        choices=["mushroom", "congressional", "csv"],
        required=True,
    )
    parser.add_argument(
        "--theta",
        type=float,
        help="Theta parameter for the algorithm",
        required=True,
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Number of clusters to find",
        required=True,
    )
    parser.add_argument(
        "--approx_fn",
        type=str,
        help="Approximation function to use",
        choices=["rational_add", "rational_sub", "rational_exp", "rational_sin"],
        default="rational_sub",
    )
    parser.add_argument(
        "--split_train",
        type=float,
        help="Split train percentage",
        default=0.15,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the .csv dataset (only if dataset=csv)",
        default=None,
    )
    parser.add_argument(
        "--calculate_metrics",
        type=bool,
        help="Whether calculate metrics and runtime",
        default=True,
    )
    parser.add_argument(
        "--skip_outliers",
        help="Whether include outliers in the results",
        action="store_true",
    )
    args = parser.parse_args()
    main(args=args)
