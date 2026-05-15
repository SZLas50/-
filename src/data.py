import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = Path("data/loan_defaults.csv")


def parse_work_year(work_year: str) -> float:
    if pd.isna(work_year):
        return 0.0

    text = str(work_year).strip()
    if text == "< 1 year":
        return 0.0
    if text == "10+ years":
        return 10.0

    try:
        return float(text.split()[0])
    except (ValueError, IndexError):
        return 0.0


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load the Northern bank loan dataset from a CSV file."""
    data_path = Path(path) if path else DEFAULT_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    return pd.read_csv(data_path)


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess the Northern bank loan dataset for model training."""
    df = df.copy()
    if "isDefault" not in df.columns:
        raise ValueError("输入数据必须包含 'isDefault' 列。")

    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    df["issue_year"] = df["issue_date"].dt.year.fillna(0).astype(int)
    df["issue_month"] = df["issue_date"].dt.month.fillna(0).astype(int)

    if "work_year" in df.columns:
        df["work_year_num"] = df["work_year"].apply(parse_work_year)

    categorical_cols = [
        "class",
        "employer_type",
        "industry",
        "use",
        "region",
        "initial_list_status",
        "app_type",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    feature_cols = [
        "total_loan",
        "year_of_loan",
        "interest",
        "monthly_payment",
        "work_year_num",
        "house_exist",
        "censor_status",
        "debt_loan_ratio",
        "del_in_18month",
        "scoring_low",
        "scoring_high",
        "known_outstanding_loan",
        "known_dero",
        "pub_dero_bankrup",
        "recircle_b",
        "recircle_u",
        "early_return",
        "early_return_amount",
        "early_return_amount_3mon",
        "issue_year",
        "issue_month",
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]
    feature_cols += [col for col in categorical_cols if col in df.columns]

    df = df[feature_cols + ["isDefault"]]

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "isDefault"]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], drop_first=True, dtype=int)

    y = df.pop("isDefault")
    X = df
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into训练集和测试集."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y) # type: ignore
