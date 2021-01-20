import pandas as pd # pandasのインポート
# irisファイルを読み込んで、データフレームに変換
df = pd.read_csv("iris.csv")
print(df.head())

# uniqueメソッドで種類列の値を確認
print(df["種類"].unique())
syurui = df["種類"].unique()

# array型の特定要素を参照
print(syurui[0])

# value_countsメソッドでデータの出現回数をカウント
print(df["種類"].value_counts())

print(df.tail(3))

# isnullメソッドで欠損値の有無を調べる
print(df.isnull()) #各マスが欠損値かどうかを調べる

# anyメソッドにより列単位で欠損値を確認
print(df.isnull().any(axis = 0))

# sumメソッドで各列の合計値を求める
print(df.sum()) # 各列の合計値を計算

# isnullメソッドとsumメソッドで各列の欠損値の数を求める
tmp = df.isnull()
print(tmp.sum())

# dropnaメソッドで欠損値を含む行/列を削除する
# 欠損値が一つでもある行を削除した結果をdf2に代入
df2 = df.dropna(how = "any" , axis = 0)
print(df2.tail())

# 削除元のデータフレームを確認
print(df.isnull().any(axis = 0))

# fillnaメソッドで欠損値を指定した値に置き換える
df["花弁長さ"] = df["花弁長さ"].fillna(0)
print(df.tail())

# meanメソッドで数値の列の平均値を計算
# 数値列の各平均値を計算（文字列の列は自動的に除外してくれる）
print(df.mean())

# 特定の列だけを計算する
# 「がく片長さ」列の平均値を計算
print(df["がく片長さ"].mean())

# 各列の標準偏差
print(df.std())

# 平均値を求めてデーフレームの欠損値と置き換える
df = pd.read_csv("iris.csv")

# 各列の平均値を計算してcolumnsに代入
colmean = df.mean()

# 平均値で欠損値を穴埋めしてdf2に代入
df2 = df.fillna(colmean)

# 欠損値があるか確認
print(df2.isnull().any(axis = 0))
