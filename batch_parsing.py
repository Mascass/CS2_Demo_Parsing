from demoparser2 import DemoParser
from pathlib import Path
import os

DEMOS_DIR = Path("demos/")

EVENTS = {
    "kills":   ("player_death", ["team_name", "X", "Y"]),
    "damages":  ("player_hurt",  ["team_name", "X", "Y"]),
    "rounds": ("round_end", []),
    "chat": ("chat_message", []),
}
TICK_PROPS = ["X", "Y", "Z", "pitch", "yaw", "health", "armor_value",
              "team_name","player_name", "is_alive", "current_equip_value"]

def batch_parsing():
    for dem_path in DEMOS_DIR.glob("*.dem"):
        match_name = dem_path.stem
        output_dir = f"parsed/{match_name}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Parsing {match_name}...")
        parser = DemoParser(str(dem_path))

        # Parse events
        for name, (event, props) in EVENTS.items():
            df = parser.parse_event(event, player=props)
            df.to_parquet(f"{output_dir}/{name}.parquet")

        # Parse only ticks on death
        wanted_ticks = parser.parse_event("player_death")["tick"].tolist()
        ticks_df = parser.parse_ticks(TICK_PROPS,ticks=wanted_ticks)
        ticks_df.to_parquet(f"{output_dir}/ticks.parquet")
    print("batch_parsing Done")


batch_parsing()