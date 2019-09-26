import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import statistics as st
from matplotlib.colors import ListedColormap
import seaborn as sns


#Load the data
df = pd.read_csv("train.csv")

#Quick data exploration
print(df.head(10))
summ = df.describe()


