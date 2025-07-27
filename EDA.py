# importing pandas, and other necessary modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

df = pd.read_csv("CodeAndData/Data/Training.csv")

# select numerical features
numerical_df = df.select_dtypes(include=[np.number])

# callculate correlation exluding status
correlations = numerical_df.corr()['Status'].drop('Status').drop('Stage')

# get top 3
top = correlations.abs().sort_values(ascending=False).head(3)

print(top)

# calculate num of bins using square root rule
bin = int(np.sqrt(334))

# histogram plots
sns.histplot(df, x="Bilirubin", hue="Status", bins=bin)
plt.show()
sns.histplot(df, x="Prothrombin", hue="Status", bins=bin)
plt.show()
sns.histplot(df, x="Copper", hue="Status", bins=bin)
plt.show()
sns.histplot(df["Status"], bins=bin, discrete=True)
plt.show()

# checking null values
print(df.isnull().sum())
