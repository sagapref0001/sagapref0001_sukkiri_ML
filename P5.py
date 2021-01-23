# pandasライブラリをインポート
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

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

x = df2[xcol]
t = df2["target"]

# 学習に利用するデータと予測性能の検証用データに分割（検証用20%、整数０を用いて乱数シード)
x_train, x_test, y_train, y_test = train_test_split(x,t, test_size = 0.2, random_state = 0)

# 訓練データで学習、木の最大の深さは３、乱数シードは０
model = tree.DecisionTreeClassifier(max_depth = 3, random_state = 0)
model.fit(x_train, y_train)

# 学習済モデルに対してテストデータでの正答率を計算
print(model.score(x_test,y_test))

new_data = [[1.56,0.23,-1.1,-2.8]]

print(model.predict(new_data))
