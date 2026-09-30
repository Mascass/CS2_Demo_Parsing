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


def clean(player_df):
    player_df = player_df.drop(columns=player_df.filter(like="name").columns)
    player_df = player_df.drop(columns=player_df.filter(like="assister").columns)
    player_df = player_df.drop(columns="hitgroup")
    player_df = player_df.drop(columns="dominated")
    player_df = player_df.drop(columns=player_df.filter(like="X").columns)
    player_df = player_df.drop(columns=player_df.filter(like="Y").columns)
    player_df = player_df.drop(columns=["weapon","weapon_originalowner_xuid"])
    return player_df


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


def add_pair_key(df, a, b):
    df = df.copy()
    df[a] = df[a].astype(float)
    df[b] = df[b].astype(float)
    df["p_lo"] = np.minimum(df[a], df[b])   # order-independent pair id
    df["p_hi"] = np.maximum(df[a], df[b])
    return df


def get_first_damage_ticks():
    if PARSED_DIR.glob("*/kills.parquet"):
        extract_kills_df = pd.DataFrame()
        for demo_path in PARSED_DIR.glob("*/"):
            dmg_df = pd.read_parquet(os.path.join(demo_path, "damages.parquet"))
            kills_df = bp.process_duel(demo_path).reset_index(names="kill_id")
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

        return extract_kills_df
    else:
        return FileNotFoundError


def parse_sample_ticks(sample_ticks):
    ticks_props = ["X", "Y", "Z", "pitch", "yaw", "health", "armor_value",
                  "team_name", "is_alive", "current_equip_value"]
    wanted_df = pd.DataFrame()
    for demo_path in Path("demos/").glob("*.dem"):
        match_name = demo_path.stem
        parser = DemoParser(str(demo_path))
        wanted_ticks = sample_ticks[sample_ticks["match_name"] == match_name]
        wanted_ticks = wanted_ticks["sample_tick"].tolist()
        ticks_df = parser.parse_ticks(ticks_props,ticks=wanted_ticks)
        ticks_df["match_name"] = match_name

        wanted_df = pd.concat([wanted_df,ticks_df], ignore_index=True)
    wanted_df.to_parquet("data/prediction_ticks.parquet")
    print("prediction_ticks.parquet Done")


def set_data():
    data = get_first_damage_ticks()
    print(data["sample_tick"])
    print(data.columns)
    # data.to_csv("data/ml_kills_data.csv")
    parse_sample_ticks(data[["sample_tick","match_name"]])
    # Need to create a match_id instead of match_name to use in a larger dataset and not have duplicates

def get_dataset():
    player_data = pd.read_parquet("data/prediction_ticks.parquet")
    duel_data = pd.read_csv("data/ml_kills_data.csv")
    player_data = player_data.rename(columns={"tick": "sample_tick"})

    killer_data = player_data.merge(duel_data[["sample_tick","match_name","attacker_steamid"]].drop_duplicates(),
                                    left_on=["sample_tick","match_name","steamid"],
                                    right_on=["sample_tick","match_name","attacker_steamid"]
                                    )

    victim_data = player_data.merge(duel_data[["sample_tick","match_name","user_steamid"]].drop_duplicates(),
                                    left_on=["sample_tick","match_name","steamid"],
                                    right_on=["sample_tick","match_name","user_steamid"]
                                    )

    killer_data.drop(columns=["attacker_steamid"], inplace=True)
    victim_data.drop(columns=["user_steamid"], inplace=True)
    with pd.option_context("display.max_columns", None):
        print(killer_data.head(10))
    print(killer_data.columns)
    with pd.option_context("display.max_columns", None):
        print(victim_data.head(10))
    print(victim_data.columns)

# set_data()
get_dataset()