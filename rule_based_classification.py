import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#define two functions

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

# TASK 1:

# Q1: Read persona.csv and show general information about the dataset.

df = pd.read_csv("Hafta 2/ödevler/persona.csv")
check_df(df)

# Q2: How many unique "source" are there? What are their frequencies?

df["SOURCE"].nunique()
# df["SOURCE"].unique()
df["SOURCE"].value_counts()

# Q3: How many unique "price" are there?

df["PRICE"].nunique()
# df["SOURCE"].unique()

# Q4: How many sales were made from which price?

cat_summary(df, "PRICE")
# df["PRICE"].value_counts()

# Q5: How many sales from which country?

df["COUNTRY"].value_counts()

# Q6: How much was earned in total from sales by country?

df.groupby("COUNTRY")["PRICE"].agg(["sum"])
# df.groupby("COUNTRY").agg({"PRICE":"sum"})

# Q7: What are the sales numbers according to SOURCE types?

df["SOURCE"].value_counts()

# Q8: What are the PRICE averages by country?

df.groupby("COUNTRY")["PRICE"].agg(["mean"])
# df.groupby("COUNTRY").agg({"PRICE":"mean"})

# Q9: What are the PRICE averages by SOURCEs?

df.groupby("SOURCE")["PRICE"].agg(["mean"])
# df.groupby("SOURCE").agg({"PRICE":"mean"})

# Q10: What are the PRICE averages in the COUNTRY-SOURCE breakdown?

df.groupby(["COUNTRY", "SOURCE"])["PRICE"].agg(["mean"])
# df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE":"mean"})

# TASK 2:

# What are the total gains broken down by COUNTRY, SOURCE, SEX, AGE?

df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])["PRICE"].agg(["sum"])
# df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "sum"})

# TASK 3:

# Sort the output by PRICE.

agg_df = df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE": "sum"}).sort_values(by="PRICE", ascending= False)
agg_df.head()

# TASK 4:

# Convert the names in the index to variable names.

agg_df = agg_df.reset_index()
agg_df.head()

# TASK 5:

# Convert age variable to categorical variable and add it to agg_df.

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=[0, 19, 24, 31, 41, 70],labels=['0_18','19_23','24_30','31_40','41_70'])
agg_df.head()

# TASK 6:

# Identify new level-based customers (personas).

agg_df.values
agg_df["customers_level_based"] = [row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]
agg_df.head()

persona_df = agg_df[["customers_level_based", "PRICE"]]
persona_df = persona_df.groupby(by="customers_level_based").agg("mean")
persona_df = persona_df.reset_index()
persona_df

# TASK 7:

# Segment your new customers (personas).
# Divide new customers (Example: USA_ANDROID_MALE_0_18) into 4 segments according to PRICE.
# Add the segments to agg_df as variable with SEGMENT naming.
# Describe the segments (Group by segments and get the price mean, max, sum).
# Analyze C segment (only extract C segment from dataset and analyze).

persona_df["SEGMENT"] = pd.qcut(persona_df["PRICE"], 4, labels=["D", "C", "B", "A"])
persona_df.head()

persona_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})

persona_df[persona_df["SEGMENT"] == "C"].describe().T
agg_df = persona_df
agg_df

# TASK 8:

# What segment does a 33-year-old Turkish woman using ANDROID belong to, and how much income is expected on average?

new_user1 = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user1]

# What segment does a 35-year-old French woman using IOS belong to, and how much income is expected on average?

new_user2 = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user2]