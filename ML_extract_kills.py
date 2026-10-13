from pathlib import Path
from winreg import error

import errno
import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
from demoparser2 import DemoParser

from util import batch_processing as bp

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
            kills_df = pd.read_parquet(os.path.join(demo_path, "kills.parquet")).reset_index(names="kill_id")
            # Keep the below columns in the dataset to merge with damage and tick info later
            kills_df = kills_df[["attacker_steamid","user_steamid","tick","kill_id"]]
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

        extract_kills_df.to_csv("data/ml_kills_data.csv")
        print("ml_kills_data.csv Done")
        return extract_kills_df
    else:
        return FileNotFoundError


''' Clean dataset to fit the wanted columns of the prediction dataset'''
def clean_prediction_dataset(wanted_df):
    try:
        dummies = pd.get_dummies(wanted_df['team_name'])
        wanted_df = pd.concat([wanted_df, dummies], axis=1)
        wanted_df = wanted_df.drop(columns=['team_name','TERRORIST'])
    except:
        "No team name column in clean_prediction_dataset"
    try:
        wanted_df = wanted_df.rename(columns={"tick": "sample_tick"})
    except:
        "No tick column in clean_prediction_dataset"
    return wanted_df

'''Parse ticks if needed or retrieve parquet file from /parsed '''
def parse_ticks_cached(parser, props, ticks, parsed_path):
    parsed_path = Path(parsed_path)
    ticks = list(ticks)

    if parsed_path.exists():
        cached = pd.read_parquet(parsed_path)
        # only reuse the cache if it covers everything we need now
        if set(ticks).issubset(cached["tick"]) and set(props).issubset(cached.columns):
            return cached[cached["tick"].isin(ticks)].reset_index(drop=True)

    df = parser.parse_ticks(props, ticks=ticks)
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parsed_path)
    return df

'''Parse ticks parquet for all demos to create a df with info on only the wanted ticks
Add match name column
'''
def parse_sample_ticks(sample_ticks):
    ticks_props = ["X", "Y", "Z", "pitch", "yaw", "health", "armor_value",
                  "team_name", "current_equip_value"]
    wanted_df = pd.DataFrame()
    wanted_df_minus_1 = pd.DataFrame()
    for demo_path in Path("demos/").glob("*.dem"):
        match_name = demo_path.stem
        parser = DemoParser(str(demo_path))
        # Parse only wanted ticks and use to get ticks-1 to calculate velocity after
        wanted_ticks = sample_ticks[sample_ticks["match_name"] == match_name]
        wanted_ticks_minus_1 = (wanted_ticks["sample_tick"]-1).tolist()
        wanted_ticks = wanted_ticks["sample_tick"].tolist()

        parsed_path = Path("parsed") / match_name
        ticks_df = parse_ticks_cached(parser, ticks_props, wanted_ticks, parsed_path / "sample_ticks.parquet")
        ticks_df_minus_1 = parse_ticks_cached(parser, ["X", "Y", "Z"], wanted_ticks_minus_1,
                                              parsed_path / "sample_ticks_minus_1.parquet")

        # ticks_df = parser.parse_ticks(ticks_props,ticks=wanted_ticks)
        # ticks_df_minus_1 = parser.parse_ticks(["X","Y","Z"],ticks=wanted_ticks_minus_1)
        ticks_df["match_name"] = match_name
        ticks_df_minus_1["match_name"] = match_name

        wanted_df = pd.concat([wanted_df,ticks_df], ignore_index=True)
        wanted_df_minus_1 = pd.concat([wanted_df_minus_1,ticks_df_minus_1], ignore_index=True)


    # Clean the dataset as wanted to get the best prediction dataset
    wanted_df = clean_prediction_dataset(wanted_df)
    wanted_df.to_parquet("data/prediction_ticks.parquet")
    wanted_df_minus_1.to_parquet("data/velocity_calc_ticks.parquet")
    print("prediction_ticks.parquet Done")

'''df : prediction_dataset'''
def get_velocity(dataset,nb_of_ticks=1):
    velocity_df = pd.read_parquet("data/velocity_calc_ticks.parquet")
    # Get nb_of_ticks previous ticks (need to parse all ticks for this to actually work)
    dataset["prev_tick"] = dataset["sample_tick"] - nb_of_ticks
    # Merge
    velocity_props = ["tick", "steamid", "X", "Y", "Z"]
    pred_propsK = ["prev_tick", "K_steamid", "K_X", "K_Y", "K_Z"]
    pred_propsV = ["prev_tick", "V_steamid", "V_X", "V_Y", "V_Z"]
    # Merge only killers
    df_k = (
        velocity_df[velocity_props].merge(dataset[pred_propsK], right_on=["prev_tick", "K_steamid"],
               left_on=["tick", "steamid"], how="right")
    ).dropna()
    # Merge only victims
    df_v = (
        velocity_df[velocity_props].merge(dataset[pred_propsV], right_on=["prev_tick", "V_steamid"],
               left_on=["tick", "steamid"], how="right")
    ).dropna()

    # Velocity computation for each
    df_k["velocity"] = np.sqrt(
        (df_k["K_X"] - df_k["X"]) ** 2 +
        (df_k["K_Y"] - df_k["Y"]) ** 2 +
        (df_k["K_Z"] - df_k["Z"]) ** 2
    )*64/nb_of_ticks
    df_v["velocity"] = np.sqrt(
        (df_v["V_X"] - df_v["X"]) ** 2 +
        (df_v["V_Y"] - df_v["Y"]) ** 2 +
        (df_v["V_Z"] - df_v["Z"]) ** 2
    )*64/nb_of_ticks
    # Merge on the same row velocity_x = K_velocity and velocity_y = V_velocity
    df = (
        dataset[["prev_tick", "K_steamid","V_steamid"]]
        .merge(df_k[["prev_tick", "K_steamid", "velocity"]], on=["prev_tick", "K_steamid"], how="left")
        .merge(df_v[["prev_tick", "V_steamid", "velocity"]], on=["prev_tick", "V_steamid"], how="left")
    ).drop_duplicates()

    return df["velocity_x"], df["velocity_y"]

'''Create a csv file extracting kills to use in a prediction with the wanted columns. '''
def set_data():
    data = get_first_damage_ticks()
    parse_sample_ticks(data[["sample_tick","match_name"]])
    # Need to create a match_id instead of match_name to use in a larger dataset and not have duplicates


'''Create prediction dataset using df created previously in set_data()'''
def get_dataset():
    player_data = pd.read_parquet("data/prediction_ticks.parquet")
    duel_data = pd.read_csv("data/ml_kills_data.csv")
    # Separate into killer and victim dataframe then merge them together (steam_id to fix dupes and bugs) to get kill dataset
    killer_data = player_data.merge(duel_data[["sample_tick","match_name","attacker_steamid"]].drop_duplicates(),
                                    left_on=["sample_tick","match_name","steamid"],
                                    right_on=["sample_tick","match_name","attacker_steamid"]
                                    )

    victim_data = player_data.merge(duel_data[["sample_tick","match_name","user_steamid","attacker_steamid"]].drop_duplicates(),
                                    left_on=["sample_tick","match_name","steamid"],
                                    right_on=["sample_tick","match_name","user_steamid"]
                                    )

    state_cols = ["sample_tick", "match_name"]
    killer_data = killer_data.rename(columns=lambda c: f"K_{c}" if c not in state_cols else c)
    victim_data = victim_data.rename(columns=lambda c: f"V_{c}" if c not in state_cols else c)
    training_data = killer_data.merge(victim_data,
                                      left_on=["sample_tick", "match_name","K_steamid"],
                                      right_on=["sample_tick", "match_name","V_attacker_steamid"]
                                      )

    # A is equivalent to K at this point
    training_data["A_dies"] = False

    # Create velocity column
    training_data["K_velocity"],training_data["V_velocity"] = get_velocity(training_data,nb_of_ticks=1)

    dataset = training_data.copy()
    dataset["kill_id"] = (dataset["sample_tick"].astype(str) + "_" + dataset["match_name"].astype(str) + "_"
                          + dataset["K_steamid"].astype(str) + "_" + dataset["V_steamid"].astype(str))
    training_data.drop(columns=["sample_tick","K_attacker_steamid","V_user_steamid","V_attacker_steamid"], inplace=True)

    # Create randomized swap of K and V
    rng = np.random.default_rng()
    swap = rng.random(len(training_data)) < 0.5
    keeping_cols = ["match_name","kill_id","K_steamid","V_steamid"]
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
    # Create A_sees_B and B_sees_A columns
    a_pos = (dataset["A_X"], dataset["A_Y"], dataset["A_Z"])
    b_pos = (dataset["B_X"], dataset["B_Y"], dataset["B_Z"])
    dataset["A_sees_B"] = bp.is_looking_at(a_pos, dataset["A_pitch"], dataset["A_yaw"], b_pos)
    dataset["B_sees_A"] = bp.is_looking_at(b_pos, dataset["B_pitch"], dataset["B_yaw"], a_pos)
    # Create distance between 2 players column
    diff = np.array([
        dataset["A_X"] - dataset["B_X"],
        dataset["A_Y"] - dataset["B_Y"],
        dataset["A_Z"] - dataset["B_Z"],
    ])
    dataset["distance"] = np.linalg.norm(diff, axis=0).round(1)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print("0 distance kills",dataset[dataset["distance"] == 0.])
    dataset = dataset[dataset["distance"] != 0.] # Removes self kills

    # Create difference columns
    diff_cols = ["health", "armor_value", "current_equip_value", "Z"]
    for d in diff_cols:
        a_col, b_col = f"A_{d}", f"B_{d}"
        if a_col in dataset.columns and b_col in dataset.columns:
            dataset[f"diff_{d}"] = dataset[a_col] - dataset[b_col]

    print(dataset.columns)
    # Drop columns ruining the prediction
    for d in diff_cols:
        a_col, b_col = f"A_{d}", f"B_{d}"
        dataset.drop(columns=[a_col, b_col], inplace=True)
    dataset.drop(columns=["A_X","A_Y","B_X","B_Y","A_pitch","A_yaw","B_pitch","B_yaw","Unnamed: 0"], inplace=True)
    dataset.to_csv("data/adjusted_prediction_dataset.csv")
    print("prediction_dataset.csv Adjusted")


set_data()
get_dataset()
adjust_dataset()