from pathlib import Path
import polars as pl
import pandas as pd

PARSED_DIR = Path("parsed/")

# Processing kills.parquet files
def process_kills():
    if PARSED_DIR.glob("*/kills.parquet"):
        for parquet_path in PARSED_DIR.glob("*/kills.parquet"):
            print(parquet_path)
            kills   = pl.read_parquet(parquet_path)
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

# Processing chat.parquet files
def process_chat():
    if PARSED_DIR.glob("*/chat.parquet"):
        for parquet_path in PARSED_DIR.glob("*/chat.parquet"):
            print(parquet_path)
            chat = pl.read_parquet(parquet_path)
            df_chat = chat.to_pandas()

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



process_kills()
process_chat()