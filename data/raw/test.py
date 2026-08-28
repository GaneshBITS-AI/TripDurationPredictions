import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train.csv')
df2 = df.head(100000)
df2.to_csv('NYC.csv', index=False)