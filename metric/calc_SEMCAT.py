import argparse
import pandas as pd

TARGET_METRICS = ["ESMATCH++mac", "RoSE3-99"]

def ensure_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    'index' 컬럼을 보장:
    - 헤더가 비어 있거나('') 혹은 'Unnamed: 0' 류면 'index'로 변경
    - 'index'를 포함하는 이름이 있으면 그것을 'index'로 변경
    - 위가 모두 아니면 df.index 를 꺼내 새 'index' 컬럼으로 추가
    """
    # 열 이름 정리 (NaN -> '', strip)
    cols = [("" if (isinstance(c, float) and pd.isna(c)) else str(c).strip()) for c in df.columns]
    df.columns = cols

    if "index" in df.columns:
        return df

    # 빈 헤더("")를 index로 사용
    if "" in df.columns:
        df = df.rename(columns={"": "index"})
        return df

    # 'Unnamed: 0', 'Unnamed: 0_level_0' 등 처리
    for c in df.columns:
        if c.lower().startswith("unnamed"):
            df = df.rename(columns={c: "index"})
            return df

    # 'index' 라는 단어를 포함하는 컬럼명
    for c in df.columns:
        if "index" in c.lower():
            df = df.rename(columns={c: "index"})
            return df

    # 모두 아니면 df.index 를 컬럼으로 꺼냄
    df = df.reset_index().rename(columns={"index": "index"})
    return df

def build_semcat(df: pd.DataFrame, alpha: float, require_both: bool) -> pd.DataFrame:
    """
    SEMCAT = w1 * ESMATCH++mac + w2 * RoSE3-99
    where raw weights are w1_raw = 1, w2_raw = (1 - alpha), then normalized.
    alpha in [0, 1].
    """
    required_cols = {"index", "name", "metric", "item", "score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha는 0과 1 사이여야 합니다. 예: 0.25")

    dft = df[df["metric"].isin(TARGET_METRICS)].copy()
    dft["score"] = pd.to_numeric(dft["score"], errors="coerce")

    grouped = (
        dft.groupby(["index", "name", "item", "metric"], as_index=False)["score"]
        .mean()
    )

    pivot = grouped.pivot_table(
        index=["index", "name", "item"],
        columns="metric",
        values="score",
        aggfunc="first",
    )

    if require_both:
        pivot = pivot.dropna(subset=TARGET_METRICS, how="any")

    pivot = pivot.reindex(columns=TARGET_METRICS).fillna(0.0)

    w1_raw = 1.0
    w2_raw = 1.0 - float(alpha)
    total = w1_raw + w2_raw
    w1 = w1_raw / total if total else 0.0
    w2 = w2_raw / total if total else 0.0

    semcat = w1 * pivot["ESMATCH++mac"] + w2 * pivot["RoSE3-99"]
    semcat_df = semcat.rename("score").reset_index()
    semcat_df["metric"] = "SEMCAT"
    semcat_df = semcat_df[["index", "name", "metric", "item", "score"]]

    return pd.concat([df, semcat_df], ignore_index=True)

def main():
    parser = argparse.ArgumentParser(description="SEMCAT metric 생성(가중치 = 1 과 (1-a))")
    parser.add_argument("--in", dest="in_csv", default="./result/per-item.csv", help="입력 CSV 경로")
    parser.add_argument("--out", dest="out_csv", default="./result/per-item_SEMCAT.csv", help="출력 CSV 경로")
    parser.add_argument("--alpha", type=float, default=0.25, help="(1-a) 에서 a 값 (0~1)")
    parser.add_argument("--require-both", action="store_true", help="두 metric이 모두 있는 항목만 계산")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    df = ensure_index_column(df)  # ← 여기서 'index' 보장
    out_df = build_semcat(df, alpha=args.alpha, require_both=args.require_both)
    out_df.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
