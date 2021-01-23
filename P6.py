import pandas as pd

df = pd.read_csv("ex3.csv")
print(df.head())

# 欠損値があるかの確認
print(df.isnull().any(axis=0))

# 欠損値を中央値で穴埋め
df2 = df.fillna(df.median())
print(df2.isnull().any(axis=0))

# # 各特徴量x0～x3とtarget列との散布図を作成し、外れ値があるか検討
# % matplotlib
# inline
#
# df2.plot(kind="scatter", x="x0", y="target")
# df2.plot(kind="scatter", x="x1", y="target")
# df2.plot(kind="scatter", x="x2", y="target")
# df2.plot(kind="scatter", x="x3", y="target")

# 2つの条件で外れ値の行を特定する
print(df[(df["x2"] < -2) & (df["target"] > 100)])

# 外れ値を含む行の削除
df3 = df2.drop(23, axis = 0)
print(df3.shape)

# スライス構文でdf3から特徴量の変数ｘと正解データの変数ｔに分割
x = df3.loc[ : , "x0":"x3"] #特徴量の取り出し
t = df3["target"] #正解データの取り出し

#　訓練データとテストデータに分割する
from sklearn.model_selection import train_test_split
x_train, x_test, y_train ,y_test = train_test_split(x, t, test_size=0.2, random_state = 1)

# 重回帰モデルのLineRegression関数をインポートする
from sklearn.linear_model import LinearRegression

# LinearRegression関数を使ってモデルを作成する
model = LinearRegression()
model.fit(x_train, y_train)

# 決定係数を計算し予測性能が良いかどうか判断する
print(model.score(x_test, y_test))


