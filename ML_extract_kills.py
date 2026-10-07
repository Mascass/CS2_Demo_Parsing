from pathlib import Path
import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
from demoparser2 import DemoParser

import util.util_processing as util
import batch_processing as bp

PARSED_DIR = Path("parsed/")

'''Remove unwanted columns for correlation matrix'''
def clean(player_df):
    player_df = player_df.drop(columns=player_df.filter(like="name").columns)
    player_df = player_df.drop(columns=player_df.filter(like="assister").columns)
    player_df = player_df.drop(columns="hitgroup")
    player_df = player_df.drop(columns="dominated")
    player_df = player_df.drop(columns=player_df.filter(like="X").columns)
    player_df = player_df.drop(columns=player_df.filter(like="Y").columns)
    player_df = player_df.drop(columns=["weapon","weapon_originalowner_xuid"])
    return player_df

'''Correlation matrix of kills data'''
def predict_corr():
    try:
        data_df = pd.read_csv("data/ml_kills_data.csv")
    except FileNotFoundError:
        print("File not found.")
        return
    data_df = clean(data_df)
    corr = data_df.corr()
    # order = corr[" "].sort_values(ascending=False).index
    # corr_ordered = corr.loc[order, order]
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(20, 16))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.show()

'''attacker and victim pair key'''
def add_pair_key(df, a, b):
    df = df.copy()
    df[a] = df[a].astype(float)
    df[b] = df[b].astype(float)
    df["p_lo"] = np.minimum(df[a], df[b])   # order-independent pair id
    df["p_hi"] = np.maximum(df[a], df[b])
    return df

'''Parse kills and damage parquet for all demos to return a df witht the tick from the first damage of a kills (4 ticks before)'''
def get_first_damage_ticks():
    if PARSED_DIR.glob("*/kills.parquet"):
        extract_kills_df = pd.DataFrame()
        for demo_path in PARSED_DIR.glob("*/"):
            dmg_df = pd.read_parquet(os.path.join(demo_path, "damages.parquet"))
            kills_df = bp.process_duel(demo_path).reset_index(names="kill_id")
            # Keep the below columns in the dataset to merge with damage and tick info later
            kills_df = kills_df[["attacker_steamid","user_steamid","tick","kill_id","is_duel"]]
            kill_p = add_pair_key(kills_df, "attacker_steamid", "user_steamid")
            hurt_p = add_pair_key(dmg_df, "attacker_steamid", "user_steamid")

            WINDOW = 3 * 64  # look back at most 3 s (64 tick/s)

            m = kill_p[["kill_id", "tick", "p_lo", "p_hi"]].merge(
                hurt_p[["tick", "p_lo", "p_hi"]], on=["p_lo", "p_hi"], suffixes=("_kill", "_hurt"))
            m = m[(m["tick_hurt"] <= m["tick_kill"]) & (m["tick_hurt"] >= m["tick_kill"] - WINDOW)]

            first_hit = m.groupby("kill_id")["tick_hurt"].min().rename("first_hit_tick")

            kills_df = kills_df.merge(first_hit, on="kill_id", how="inner")  # drops kills with no prior hit
            kills_df["sample_tick"] = kills_df["first_hit_tick"] - 4  # a few ticks BEFORE the hit
            match_name = demo_path.stem
            kills_df["match_name"] = match_name
            extract_kills_df = pd.concat([extract_kills_df, kills_df], ignore_index=True)

        return extract_kills_df
    else:
        return FileNotFoundError


''' Clean dataset to fit the wanted columns of the prediction dataset'''
def clean_prediction_dataset(wanted_df):
    dummies = pd.get_dummies(wanted_df['team_name'])
    wanted_df = pd.concat([wanted_df, dummies], axis=1)
    wanted_df = wanted_df.drop(columns=['team_name','TERRORIST'])
    wanted_df = wanted_df.rename(columns={"tick": "sample_tick"})
    return wanted_df


'''Parse ticks parquet for all demos to create a df with info on only the wanted ticks
Add match name column
'''
def parse_sample_ticks(sample_ticks):
    ticks_props = ["X", "Y", "Z", "pitch", "yaw", "health", "armor_value",
                  "team_name", "current_equip_value"]
    wanted_df = pd.DataFrame()
    for demo_path in Path("demos/").glob("*.dem"):
        match_name = demo_path.stem
        parser = DemoParser(str(demo_path))
        wanted_ticks = sample_ticks[sample_ticks["match_name"] == match_name]
        wanted_ticks = wanted_ticks["sample_tick"].tolist()

        ticks_df = parser.parse_ticks(ticks_props,ticks=wanted_ticks)
        ticks_df["match_name"] = match_name
        wanted_df = pd.concat([wanted_df,ticks_df], ignore_index=True)


    # Clean the dataset as wanted to get the best prediction dataset
    wanted_df = clean_prediction_dataset(wanted_df)
    wanted_df.to_parquet("data/prediction_ticks.parquet")
    print("prediction_ticks.parquet Done")

'''Create a csv file extracting kills to use in a prediction with the wanted columns. '''
def set_data():
    data = get_first_damage_ticks()
    data.to_csv("data/ml_kills_data.csv")
    print("ml_kills_data.csv Done")
    parse_sample_ticks(data[["sample_tick","match_name"]])
    # Need to create a match_id instead of match_name to use in a larger dataset and not have duplicates


'''Create prediction dataset using df created previously in set_data()'''
def get_dataset():
    player_data = pd.read_parquet("data/prediction_ticks.parquet")
    duel_data = pd.read_csv("data/ml_kills_data.csv")

    killer_data = player_data.merge(duel_data[["sample_tick","match_name","attacker_steamid","is_duel"]].drop_duplicates(),
                                    left_on=["sample_tick","match_name","steamid"],
                                    right_on=["sample_tick","match_name","attacker_steamid"]
                                    )

    victim_data = player_data.merge(duel_data[["sample_tick","match_name","user_steamid","is_duel"]].drop_duplicates(),
                                    left_on=["sample_tick","match_name","steamid"],
                                    right_on=["sample_tick","match_name","user_steamid"]
                                    )
    state_cols = ["sample_tick", "match_name","is_duel"]
    killer_data = killer_data.rename(columns=lambda c: f"K_{c}" if c not in state_cols else c)
    victim_data = victim_data.rename(columns=lambda c: f"V_{c}" if c not in state_cols else c)
    training_data = killer_data.merge(victim_data, on=["sample_tick", "match_name","is_duel"])

    training_data["A_dies"] = False
    dataset = training_data.copy()
    dataset["kill_id"] = (dataset["sample_tick"].astype(str) + "_" + dataset["match_name"].astype(str) + "_"
                          + dataset["K_steamid"].astype(str) + "_" + dataset["V_steamid"].astype(str))

    training_data.drop(columns=["sample_tick","K_attacker_steamid","V_user_steamid"], inplace=True)

    rng = np.random.default_rng()
    swap = rng.random(len(training_data)) < 0.5
    keeping_cols = ["match_name","kill_id","is_duel","K_steamid","V_steamid"]
    label_col = "A_dies"
    featured_cols = [c for c in dataset.columns if c not in keeping_cols + [label_col]]
    cols_to_swap = sorted({
        c[2:] for c in training_data.columns
        if c in featured_cols and (c.startswith("K_") or c.startswith("V_"))
    })

    dataset = dataset[keeping_cols]
    dataset["A_dies"] = swap.astype(bool)
    for c in cols_to_swap:
        dataset[f"A_{c}"] = np.where(swap, training_data[f"V_{c}"], training_data[f"K_{c}"])
        dataset[f"B_{c}"] = np.where(swap, training_data[f"K_{c}"], training_data[f"V_{c}"])

    dataset.to_csv("data/prediction_dataset.csv")
    print("prediction_dataset.csv Done")


def adjust_dataset():
    dataset = pd.read_csv("data/prediction_dataset.csv")
    a_pos = (dataset["A_X"], dataset["A_Y"], dataset["A_Z"])
    b_pos = (dataset["B_X"], dataset["B_Y"], dataset["B_Z"])
    dataset["A_sees_B"] = bp.is_looking_at(a_pos, dataset["A_pitch"], dataset["A_yaw"], b_pos)
    dataset["B_sees_A"] = bp.is_looking_at(b_pos, dataset["B_pitch"], dataset["B_yaw"], a_pos)

    diff = np.array([
        dataset["A_X"] - dataset["B_X"],
        dataset["A_Y"] - dataset["B_Y"],
        dataset["A_Z"] - dataset["B_Z"],
    ])
    dataset["distance"] = np.linalg.norm(diff, axis=0).round(1)
    dataset = dataset[dataset["distance"] != 0.]
    dataset.drop(columns=["is_duel","A_X","A_Y","A_Z","B_X","B_Y","B_Z","A_pitch","A_yaw","B_pitch","B_yaw"], inplace=True)
    dataset.to_csv("data/prediction_dataset.csv")
    print("prediction_dataset.csv Adjusted")


# set_data()
get_dataset()
adjust_dataset()