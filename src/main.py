import argparse
from pathlib import Path

import pandas as pd
from data import load_data, preprocess_data, split_data
from model import compare_models, evaluate_model, load_model, save_model, train_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training and evaluation."""
    parser = argparse.ArgumentParser(description="信贷违约预测小项目")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--evaluate", action="store_true", help="评估模型")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/loan_defaults.csv",
        help="数据文件路径",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="model/joblib/model.joblib",
        help="保存/加载模型路径",
    )
    return parser.parse_args()


def ensure_directory_exists(path: Path) -> None:
    """Ensure the parent directory for a file path exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def print_model_results(results: dict[str, dict[str, float]]) -> None:
    """Print evaluation metrics for each model in a readable format."""
    for model_name, metrics in results.items():
        print(f"\n模型: {model_name}")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print("  详细分类报告:")
        print(pd.DataFrame(metrics["report"]).T) # type: ignore


def select_best_model(
    rf_model, lr_model, results: dict[str, dict[str, float]]
) -> tuple[str, object]:
    """Select the better model based on AUC value."""
    random_forest_auc = results["random_forest"]["auc"]
    logistic_regression_auc = results["logistic_regression"]["auc"]

    if random_forest_auc >= logistic_regression_auc:
        return "random_forest", rf_model

    return "logistic_regression", lr_model


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    ensure_directory_exists(model_path)

    if args.train:
        data_frame = load_data(str(data_path))
        X, y = preprocess_data(data_frame)
        X_train, X_test, y_train, y_test = split_data(X, y)

        rf_model, lr_model = train_model(X_train, y_train)
        evaluation_results = compare_models(rf_model, lr_model, X_test, y_test)
        best_model_name, best_model = select_best_model(
            rf_model, lr_model, evaluation_results
        )

        save_model(best_model, str(model_path))
        print(f"训练完成，最优模型: {best_model_name} 已保存到 {model_path}")
        print_model_results(evaluation_results)

    elif args.evaluate:
        model = load_model(str(model_path))
        data_frame = load_data(str(data_path))
        _, X_test, _, y_test = split_data(*preprocess_data(data_frame))

        metrics = evaluate_model(model, X_test, y_test)
        print(f"已加载模型: {model_path}")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(pd.DataFrame(metrics["report"]).T) # type: ignore

    else:
        print("请使用 --train 或 --evaluate 参数。")


if __name__ == "__main__":
    main()
