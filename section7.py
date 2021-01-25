# ライブラリのインポート
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
# %matplotlib inline

# CSVファイルの読み込み
df = pd.read_csv("Survived.csv")
print(df.head())

# Survived列のデータ
print(df["Survived"].value_counts())

# 欠損値を確認する
print(df.isnull().sum())
print(df.shape)

# Ager列を平均値で穴埋め
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Embarked列を最頻値で穴埋め
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# 特徴量ｘと正解データｔに分割する
# 特徴量として利用する列のリスト
col = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

x = df[col]
t = df["Survived"]

# 訓練データとテストデータに分割する
x_train, x_test, y_train, y_test = train_test_split(x, t, test_size =0.2, random_state = 0)

# X_trainのサイズ確認
print(x_train.shape)

# モデルの作成と学習
model = tree.DecisionTreeClassifier(max_depth = 5, random_state = 0, class_weight = "balanced")
model.fit(x_train, y_train)

# 決定木モデルの正解率を計算する
print(model.score(x_test, y_test))

# Learn関数を定義する
def learn(x, t, depth = 3):
    x_train, x_test, y_train, y_test = train_test_split(x, t, test_size = 0.2, random_state = 0)
    model = tree.DecisionTreeClassifier(max_depth = depth, random_state = 0, class_weight = "balanced")
    model.fit(x_train, y_train)
    score = model.score(X = x_train, y = y_train)
    score2 = model.score(X = x_test, y = y_test)
    return round(score,3),round(score2,3),model

# 木の深さによる正解率の変化を確認
for j in range(1,15): # jは木の深さ
    train_score, test_score, model = learn(x,t,depth = j)
    sentence = "訓練データの正解率{}"
    sentence2 = "テストデータの正解率{}"
    total_sentence = "深さ{}:" +sentence + sentence2
    print(total_sentence.format(j,train_score,test_score))

# age列の平均値と中央値を確認する
df2 = pd.read_csv("Survived.csv")
print(df2["Age"].mean())
print(df2["Age"].median())

# 小グループ作成の基準となる列を指定
print(df2.groupby("Survived").mean()["Age"])

# Pclass列で集計
print(df2.groupby("Pclass").mean()["Age"])

# ピボットテーブル機能を使う
print(pd.pivot_table(df2, index = "Survived", columns = "Pclass", values = "Age"))

# 引数aggfuncを使って平均値以外の統計量を求める
print(pd.pivot_table(df2, index = "Survived", columns = "Pclass", values = "Age", aggfunc = max))

# loc機能でAge列の欠損値を穴埋めする
# Age列の欠損値の行を抜き出す（欠損であればTrue）
is_null = df2["Age"].isnull()

# Pclass1に関する埋め込み
df2.loc[(df["Pclass"] ==1) & (df["Survived"] == 0) & (is_null),"Age"] = 43
df2.loc[(df["Pclass"] ==1) & (df["Survived"] == 1) & (is_null),"Age"] = 35

# Pclass2に関する埋め込み
df2.loc[(df["Pclass"] ==2) & (df["Survived"] == 0) & (is_null),"Age"] = 33
df2.loc[(df["Pclass"] ==2) & (df["Survived"] == 1) & (is_null),"Age"] = 25

# Pclass3に関する埋め込み
df2.loc[(df["Pclass"] ==3) & (df["Survived"] == 0) & (is_null),"Age"] = 26
df2.loc[(df["Pclass"] ==3) & (df["Survived"] == 1) & (is_null),"Age"] = 20

# Learn関するを使ってモデルを再学習させる
# 特徴量として利用する列のリスト
col = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
x = df2[col]
t = df2["Survived"]

for j in range(1,15):
    s1, s2, m = learn(x,t,depth=j)
    sentence = "深さ{}:訓練データの精度{}::テストデータの精度{}"
    print(sentence.format(j,s1,s2))

# groupbyメソッドを使って平均値を求める
sex = df2.groupby("Sex").mean()
print(sex["Survived"])

# # plotメソッドで棒グラフを作成する
# sex["Survived"].plot(kind = "bar")

# # モデルの再学習を行う
# # 特徴量として利用する列のリスト
# col = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex"]
#
# x = df2[col]
# t = df2["Survived"]
#
# train_score,test_score,model = learn(x,t) #学習

# get_dummies関数で文字列を数値に変換する
male = pd.get_dummies(df2["Sex"], drop_first = True)
print(male)

# drop_firstを指定しないget_dummies関数の戻り値
print(pd.get_dummies(df["Sex"]))

