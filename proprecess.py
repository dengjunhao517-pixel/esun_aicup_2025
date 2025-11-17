import pandas as pd
import sys


# ---------------------------------------------------------
# 工具函式：模擬 SQL 的 ROW_NUMBER() OVER (ORDER BY acct)
# ---------------------------------------------------------
def add_group_p(source_df):

    src_sorted = source_df.sort_values("acct").reset_index(drop=True)
    src_sorted["group_p"] = src_sorted.index + 1
    return src_sorted[["acct", "group_p"]]


# ---------------------------------------------------------
# PART 1 - NEXT_LEVEL
# ---------------------------------------------------------
def part_next_level(trans_df, source_df, source_grouped):

    # 內層 b: from_acct match source.acct
    b_join = pd.merge(
        trans_df,
        source_df,
        left_on="from_acct",
        right_on="acct",
        suffixes=("_trans", "_source")
    )
    b_pairs = b_join[["from_acct", "to_acct"]].drop_duplicates()

    b_inner = pd.merge(
        b_pairs, source_grouped, left_on="from_acct", right_on="acct"
    )
    sub_b = b_inner[["to_acct", "group_p"]]

    # 外層 join: a.from_acct = b.to_acct
    join_df = pd.merge(
        trans_df, sub_b, left_on="from_acct", right_on="to_acct",
        suffixes=("_trans", "_b")
    )

    # 調整欄位
    join_df = join_df.drop(columns=["to_acct_b"]).rename(
        columns={"to_acct_trans": "to_acct"}
    )
    join_df["flag"] = "NEXT_LEVEL"
    return join_df


# ---------------------------------------------------------
# PART 2 - FROM
# ---------------------------------------------------------
def part_from(trans_df, source_df, source_grouped):

    from_df = trans_df[trans_df["from_acct"].isin(source_df["acct"])]
    joined = pd.merge(
        from_df, source_grouped,
        left_on="from_acct", right_on="acct"
    ).drop(columns=["acct"])

    joined["flag"] = "FROM"
    return joined


# ---------------------------------------------------------
# PART 3 - TO
# ---------------------------------------------------------
def part_to(trans_df, source_df, source_grouped):

    to_df = trans_df[trans_df["to_acct"].isin(source_df["acct"])]
    joined = pd.merge(
        to_df, source_grouped,
        left_on="to_acct", right_on="acct"
    ).drop(columns=["acct"])

    joined["flag"] = "TO"
    return joined


# ---------------------------------------------------------
# PART 4 - BEFORE_LEVEL
# ---------------------------------------------------------
def part_before_level(trans_df, source_df, source_grouped):


    b_join = pd.merge(
        trans_df, source_df,
        left_on="to_acct", right_on="acct",
        suffixes=("_trans", "_source")
    )
    b_pairs = b_join[["from_acct", "to_acct"]].drop_duplicates()

    b_inner = pd.merge(
        b_pairs, source_grouped,
        left_on="from_acct", right_on="acct"
    )
    sub_b = b_inner[["to_acct", "group_p"]]

    joined = pd.merge(trans_df, sub_b, on="to_acct")
    joined["flag"] = "BEFORE_LEVEL"
    return joined


# ---------------------------------------------------------
# 主處理流程（對應整段 SQL）
# ---------------------------------------------------------
def generate_export(trans_df, source_df, output_filename):
    print(f"--- SQL 風格處理 {output_filename} ---")

    # 模擬 SQL: ROW_NUMBER() OVER (ORDER BY acct)
    source_grouped = add_group_p(source_df)

    # 模擬 UNION（非 UNION ALL）
    all_parts_raw = pd.concat([
        part_next_level(trans_df, source_df, source_grouped),
        part_from(trans_df, source_df, source_grouped),
        part_to(trans_df, source_df, source_grouped),
        part_before_level(trans_df, source_df, source_grouped)
    ], ignore_index=True)

    all_parts = all_parts_raw.drop_duplicates().reset_index(drop=True)

    # SQL: SELECT ... FROM (all_parts) a JOIN (source_grouped) b USING (group_p)
    final_join = pd.merge(all_parts, source_df, on="group_p", how="inner")

    final_output = final_join.rename(columns={"flag": "flag"}).copy()

    # 最終欄位
    output_columns = [
        "flag", "acct", "from_acct_type", "to_acct_type", "is_self_txn",
        "txn_amt", "txn_date", "txn_time", "currency_type", "channel_type"
    ]

    final_output_df = final_output[output_columns]

    # 全部轉大寫
    final_output_df.columns = [c.upper() for c in final_output_df.columns]

    # 輸出
    final_output_df.to_csv(output_filename, index=False, encoding="utf-8-sig")
    print(f"✔ 已輸出: {output_filename}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        print("讀取 CSV...")
        trans = pd.read_csv("acct_transaction.csv")
        alert = pd.read_csv("acct_alert.csv")
        predict = pd.read_csv("acct_predict.csv")

        trans["txn_amt"] = pd.to_numeric(trans["txn_amt"])

        generate_export(trans, predict, "Proprecessed_train_data_normal.csv")

        print("\n" + "=" * 40 + "\n")

        generate_export(trans, alert, "Proprecessed_train_data_alert.csv")

        print("\n🎉 全部完成！")

    except Exception as e:
        print("錯誤:", e)
        sys.exit(1)

