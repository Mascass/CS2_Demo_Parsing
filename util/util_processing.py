from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import os

""" Returns a df with kills per player from one game """
def get_game_kills(parquet_path):
    kills_df = pd.read_parquet(parquet_path)
    return kills_df["attacker_name"].value_counts().reset_index(name="kills")


""" Returns a df with deaths per player from one game """
def get_game_deaths(parquet_path):
    kills_df = pd.read_parquet(parquet_path)
    return kills_df["user_name"].value_counts().reset_index(name="deaths")


""" Returns df with kills, deahts and kill per death ratio per player from one game """
def get_game_kd(parquet_path):
    kills_df = pd.read_parquet(parquet_path)
    kills = kills_df.groupby("attacker_name").size().sort_values(ascending=False)
    deaths = kills_df.groupby("user_name").size().sort_values(ascending=False)
    print(kills,deaths)
    kd_df = pd.DataFrame({
        "kills": kills,
        "deaths": deaths,
        })
    kd_df["deaths"].fillna(1)
    kd_df["kd"] = kd_df["kills"] / kd_df["deaths"]
    kd_df = kd_df.sort_values(by="kd", ascending=False)
    return kd_df



# dfd = get_game_deaths("parsed/spirit-vs-falcons-m3-dust2/kills.parquet")
# print(dfd)
# Note one more death counted in this game, probably need to verify if the round is online when a death occurs
# or a restarted round or a disconnect