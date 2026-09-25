from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import os
import util.util_processing as util

PARSED_DIR = Path("parsed/")

######################## Duels processing ########################

""" Transforms horizontal and vertical angles to a vector """
def angle_to_vector(pitch, yaw):
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    x = np.cos(pitch_rad) * np.cos(yaw_rad)
    y = np.cos(pitch_rad) * np.sin(yaw_rad)
    z = -np.sin(pitch_rad)  # negative pitch = looking up in Source engine
    # /!\ Maybe need to add offset because 'z' might be at the feet of the player
    return np.array([x, y, z])

""" Distance between two players """
def distance_to_enemy(pos_self, pos_enemy):
    return np.array(pos_enemy) - np.array(pos_self)

""" Direction from one player to another """
def direction_to_enemy(pos_self, pos_enemy):
    vec = distance_to_enemy(pos_self, pos_enemy)
    # Normalize vector to get direction only
    if np.linalg.norm(vec) != 0:
        return vec / np.linalg.norm(vec)
    else:
        return vec

## Double Check later ##
""" Angle between two vectors 0=facing, 180=behind """
def angle_between(v1, v2):
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    # print(v1,v2)
    return np.degrees(np.arccos(dot))

""" Is the current player looking at the enemy? """
def is_looking_at(pos_self, pitch, yaw, pos_enemy):
    vec_self = angle_to_vector(pitch, yaw)
    vec_to_enemy = direction_to_enemy(pos_self, pos_enemy)
    angle = angle_between(vec_self, vec_to_enemy)
    threshold = 25
    # print(pos_self,pos_enemy)
    return angle < threshold

""" Are the two players in a duel? """
def is_duel(attacker, victim):
    a_pos = (attacker["X"], attacker["Y"], attacker["Z"])
    v_pos = (victim["X"], victim["Y"], victim["Z"])
    a_sees_b = is_looking_at(a_pos, attacker["pitch"], attacker["yaw"], v_pos)
    b_sees_a = is_looking_at(v_pos, victim["pitch"], victim["yaw"], a_pos)
    return a_sees_b and b_sees_a

""" Returns a df of amount of duels won per player """
def process_duel(demo_path):
    ticks_df = pd.read_parquet(os.path.join(demo_path, "ticks.parquet"))
    kills_df = pd.read_parquet(os.path.join(demo_path, "kills.parquet"))

    def analyze_kill(kill_event):
        ## Would be better to use steamid but have not parsed it yet
        tick = kill_event["tick"]
        attacker_name = kill_event["attacker_name"]
        victim_name = kill_event["user_name"]

        try:
            tick_data = ticks_df[ticks_df["tick"] == tick]
            attacker = tick_data[tick_data["player_name"] == attacker_name].iloc[0]
            victim = tick_data[tick_data["player_name"] == victim_name].iloc[0]
        except IndexError:
            return False
        if attacker is None or victim is None:
            return False
        return is_duel(attacker, victim)

    kills_df["is_duel"] = kills_df.apply(analyze_kill, axis=1)
    return kills_df


def process_duels():
    demo_folder = PARSED_DIR.glob("*/")
    all_duels = []
    for demo_path in demo_folder:
        print(demo_path)
        try:
            duels = process_duel(demo_path)
            duels["demo"] = demo_path
            all_duels.append(duels)
        except FileNotFoundError as e:
            print(f"Fichiers manquants pour {demo_path}: {e}")

    combined_duels = pd.concat(all_duels, ignore_index=True)
    combined_duels_2 = combined_duels[combined_duels["is_duel"]]
    duels_won = (
        combined_duels_2
        .groupby("attacker_name")
        .size()
        .sort_values(ascending=False)
    )
    duels_lost = (
        combined_duels_2
        .groupby("user_name")
        .size()
        .sort_values(ascending=False)
    )
    ratio_df = pd.DataFrame({
        "wins": duels_won,
        "losses": duels_lost
    })
    ratio_df.fillna(1)
    ratio_df["ratio"] = ratio_df["wins"] / ratio_df["losses"]
    ratio_df = ratio_df.sort_values("ratio", ascending=False)
    # Print ratio of won duels
    print(ratio_df)




######################## Processing kills.parquet files ########################
def process_kills():
    if PARSED_DIR.glob("*/kills.parquet"):
        for parquet_path in PARSED_DIR.glob("*/kills.parquet"):
            print(parquet_path)
            kills = pl.read_parquet(parquet_path)
            kd = (
                kills
                .group_by("attacker_name").agg(pl.len().alias("kills"))
                .join(
                    kills.group_by("user_name").agg(pl.len().alias("deaths")),
                    left_on="attacker_name", right_on="user_name", how="left"
                )
                .with_columns((pl.col("kills") / pl.col("deaths")).alias("kd"))
                .sort("kd", descending=True)
            )
            print(kd)
        else:
            print("No kills.parquet files found")

######################## Processing chat.parquet files ########################
def process_chat():
    if PARSED_DIR.glob("*/chat.parquet"):
        for parquet_path in PARSED_DIR.glob("*/chat.parquet"):
            print(parquet_path)
            chat = pl.read_parquet(parquet_path)
            df_chat = pd.read_parquet(parquet_path)

            chat_msgs = (
                chat.group_by("user_name").agg(pl.len().alias("chat_message"))
                .sort("chat_message",descending=True)
            )

            top_chatters = chat_msgs.head()["user_name"]
            i=1
            for chatter in top_chatters:
                print("top ",i," chatter is : ", chatter)
                chatter_chats = df_chat[df_chat["user_name"] == chatter][["chat_message","user_name"]]
                print(chatter_chats)
                i=i+1
            with pd.option_context('display.max_rows', None, 'display.width', None):
                print(df_chat[["user_name","chat_message"]])
    else:
        print("No chat.parquet files found")


""" Returns a df with total player kills """
def get_total_kills():
    if PARSED_DIR.glob("*/kills.parquet"):
        total_kills_df = pd.DataFrame()
        for parquet_path in PARSED_DIR.glob("*/kills.parquet"):
            kills_df = util.get_game_kills(parquet_path)
            total_kills_df = pd.concat([total_kills_df, kills_df], ignore_index=True)

        total_kills_df = (total_kills_df.groupby("attacker_name").
                          sum().
                          reset_index().
                          sort_values("kills", ascending=False))
        return total_kills_df
    else:
        return FileNotFoundError


#process_kills()
#process_chat()
#process_duels()
#
#df = util.get_game_kd("parsed/spirit-vs-falcons-m3-dust2/kills.parquet")
#print(df)