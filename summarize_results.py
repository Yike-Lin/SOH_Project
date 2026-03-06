import os
import glob
import argparse
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser("Summarize LSTM results (mean over exps for each test)")
    p.add_argument("--root", type=str, default="results_LSTM/XJTU-full",
                   help="结果根目录，例如 results_LSTM/XJTU-full")
    p.add_argument("--batches", type=int, nargs="*", default=None,
                   help="只统计这些 batch，例如: --batches 1 2 3 4 5；不填则统计root下所有batch*")
    p.add_argument("--out_csv", type=str, default="results_LSTM/XJTU-full/summary_test_mean.csv",
                   help="输出 csv：每个(batch,test)一行，为exp平均后的指标")
    return p.parse_args()


def safe_int_from_name(name, prefix):
    if not name.startswith(prefix):
        return None
    try:
        return int(name[len(prefix):])
    except:
        return None


def main():
    args = parse_args()

    pattern = os.path.join(args.root, "batch*", "test*", "exp*", "results.npz")
    files = glob.glob(pattern, recursive=True)
    if len(files) == 0:
        raise FileNotFoundError(f"没找到任何 results.npz: {pattern}")

    rows = []
    for fp in files:
        parts = os.path.normpath(fp).split(os.sep)
        batch_name, test_name, exp_name = parts[-4], parts[-3], parts[-2]
        b = safe_int_from_name(batch_name, "batch")
        t = safe_int_from_name(test_name, "test")
        e = safe_int_from_name(exp_name, "exp")

        if args.batches is not None and b not in args.batches:
            continue

        d = np.load(fp, allow_pickle=True)
        if "test_errors" not in d:
            continue

        mae, mape, mse, r2 = d["test_errors"].tolist()
        rows.append({
            "file": fp,
            "batch": b,
            "test_id": t,
            "exp": e,
            "MAE": float(mae),
            "MAPE": float(mape),
            "MSE": float(mse),
            "R2": float(r2),
        })

    df = pd.DataFrame(rows).sort_values(["batch", "test_id", "exp"])
    if df.empty:
        raise RuntimeError("筛选后没有任何结果可统计（检查 --batches 或目录结构）")

    metrics = ["MAE", "MAPE", "MSE", "R2"]

    # ===== 核心：每个 (batch, test) 对多个 exp 求均值 =====
    df_test_mean = df.groupby(["batch", "test_id"])[metrics].mean().reset_index()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_test_mean.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 保存 (batch,test) exp均值明细到: {args.out_csv}")
    print()

    def fmt_mean_std(x):
        return f"{x.mean():.6f} ± {x.std(ddof=0):.6f}"

    # ===== 每个 batch：对 test 的 exp均值再求 mean±std =====
    print("===== Per-batch summary (exp-mean per test, then mean ± std over tests) =====")
    for b, sub in df_test_mean.groupby("batch"):
        line = [f"batch{b}"]
        for m in metrics:
            line.append(f"{m}={fmt_mean_std(sub[m])}")
        print("  " + " | ".join(line))
    print()

    # ===== overall：对所有 (batch,test) 的 exp均值求 mean±std =====
    print("===== Overall summary (exp-mean per test, mean ± std over all tests) =====")
    line = ["overall"]
    for m in metrics:
        line.append(f"{m}={fmt_mean_std(df_test_mean[m])}")
    print("  " + " | ".join(line))
    print()

    # 可选：按 R2 排序看看最差/最好 test
    df_rank = df_test_mean.sort_values(["batch", "R2"], ascending=[True, True])
    print("===== Worst 10 (by R2) among (batch,test) after exp-mean =====")
    print(df_rank[["batch", "test_id", "R2", "MAE", "MSE"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()