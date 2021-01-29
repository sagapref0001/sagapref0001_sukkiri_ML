import pandas as pd

# ex4.csvをデータフレームとして読込
df = pd.read_csv("ex4.csv")
print(df.head())

print(df["sex"].mean())

print(df.groupby("class").mean()["score"])

print(pd.pivot_table(df, index = "class", columns = "sex", values = "score"))

dummy = pd.get_dummies(df["dept_id"],drop_first = True)
df2 = pd.concat([df,dummy], axis = 1)
df2 = df2.drop("dept_id",axis = 1)