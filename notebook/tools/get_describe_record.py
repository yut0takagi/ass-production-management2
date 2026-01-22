import pandas as pd


def get_describe_record(df, batch):
    rec = (
        df.loc[df["batch_id"] == batch]
        .describe()
        .stack()
        .rename("value")
        .reset_index()
    )
    rec["feature"] = rec["level_0"] + "_" + rec["level_1"]

    cols = ["X36", "X27","X30", "X41"]
    rec_wide = (
        df.loc[df["batch_id"] == batch, cols]
        .describe()
        .stack()
        .to_frame().T
    )
    rec_wide.columns = [f"{c}_{stat}" for c, stat in rec_wide.columns]
    return rec_wide
