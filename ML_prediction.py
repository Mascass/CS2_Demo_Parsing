from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
from demoparser2 import DemoParser

import util.util_processing as util
import batch_processing as bp

data=pd.read_parquet("data/prediction_ticks.parquet")

print(data.shape)
data.describe()
print(data.columns)

###################################  Start with Logistic Regression ####################################