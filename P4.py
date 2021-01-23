import pandas as pd

data = {
    "データベースの試験得点" : [70,72,75,80],
    "ネットワークの試験得点" : [80,85,79,92]
}

df = pd.DataFrame(data)
print(df)

df.index = ["一郎","次郎","三郎","太郎"]

print(df)

df = pd.read_csv("ex1.csv")
print(type(df))
print(df.head())
print(df.index)
print(df.columns)

col = ["x0","x2"]
print(df[col])
