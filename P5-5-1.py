# pandasライブラリをインポート
import pandas as pd

# ex2.csvデータを読み込んでデータフレームに変換、上位3件を表示
df = pd.read_csv("ex2.csv")
print(df.head(3))

# データフレームの行数と列数を確認
print(df.shape)

# target列にはどのようなデータがあるか調べる、各データがいくつあるのか集計
print(df["target"])
print(df["target"].value_counts())

# 特徴量の列に欠損値がいくつあるのか確認
print(df.isnull().sum())

# 欠損値を各列の中央値で穴埋め
colmedian = df.median()
df2 = df.fillna(colmedian)

# データフレームを特徴量と正解データに分割
xcol = ["x0","x1","x2","x3"]

x = df[xcol]
t = df2["target"]

