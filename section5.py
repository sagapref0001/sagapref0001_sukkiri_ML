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

# 特徴量と正解データを変数に代入
xcol = ["がく片長さ","がく片幅","花弁長さ","花弁幅"]

x = df2[xcol]
t = df2["種類"]

# モジュールノインポート
from sklearn import tree

# モデルの作成
model = tree.DecisionTreeClassifier(max_depth = 2 ,random_state = 0)

# モデルの学習と正解率の計算
model.fit(x,t) # モデルの学習
print(model.score(x , t)) #学習済みモデルの正解率計算

# 訓練データとテストデータに分割する
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,t, test_size = 0.3, random_state = 0)
# X_trainとY_trainが学習に利用するデータ
# X_testとY_testが検証に利用するデータ

# train_test_split関数の結果を確認
print(x_train.shape)
print(x_test.shape)

# 訓練データで学習と正解率の計算
# 訓練データの再訓練
model.fit(x_train, y_train)

# テストデータの予測結果と実際の答えが合致する正解率を計算
print(model.score(x_test, y_test))

# モデルを保存する
import pickle
with open("irismodel.pkl","wb") as f:
    pickle.dump(model,f)

# 分岐条件の列を求める
print(model.tree_.feature)

# 分岐条件のしきい値を含む配列を変えるtree_.threshold
print(model.tree_.threshold)

# リーフに到達したデータの数を返す tree_.value
# ノード番号１，３，４に到達したアヤメの種類と数
print(model.tree_.value[1]) # ノード番号１に到達したとき
print(model.tree_.value[2]) # ノード番号３に到達したとき
print(model.tree_.value[4]) # ノード番号４に到達したとき

# classes_でアヤメの種類とグループ番号の対応を調べる
# アヤメの種類とグループ番号の対応
print(model.classes_)

