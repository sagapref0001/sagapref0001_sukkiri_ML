import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
# %matplotlib inline

# csvファイルのと見込み
df = pd.read_csv("Survived.csv")
print(df.head())

# Survived列のデータ
print(df["Survived"].value_counts())

# 欠損値を確認する
print(df.isnull().sum())

# shapeでデータの行数を列数を確認
print(df.shape)

# Age列とEmbarked列の穴埋め
df["Age"] = df["Age"].fillna(df["Age"].mean())
# Embarked列を最頻値で穴埋め
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# 特徴量ｘと正解データｔに分類する
# 特徴量として利用する列のリスト
col = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
x = df[col]
t = df["Survived"]

# 訓練データとテストデータに分割する
x_train, x_test, y_train, y_test = train_test_split(x, t, test_size = 0.2, random_state = 0)
# x_trainのサイズ確認
print(x_train.shape)

# モデルの作成と学習
model = tree.DecisionTreeClassifier(max_depth = 5, random_state = 0, class_weight = "balanced")
model.fit(x_train, y_train) #学習

# 決定木モデルの正解率を計算する
print(model.score(X = x_test, y = y_test))

# learn関数を定義する
def learn(x, t, depth = 3):
    x_train, x_test, y_train, y_test = train_test_split(x,t,test_size = 0.2, random_state = 0)
    model = tree.DecisionTreeClassifier(max_depth = depth, random_state = 0, class_weight = "balanced")
    model.fit(x_train, y_train)

    score = model.score(X = x_test, y =y_test)
    score2 = model.score(X = x_test, y = y_test)
    return round(score,3),round(score2,3),model

# 木の深さによる正解率の変化を確認
for j in range(1,15): #jは木の深さ（１～14が入る）
    train_score, test_score, model = learn(x,t,depth=j)
    print(f"深さ{j}:訓練データの正解率{train_score}:テストデータの正解率{test_score}")

# Age列の平均値と中央値を確認する
df2 = pd.read_csv("Survived.csv")
print(df2["Age"].mean())
print(df2["Age"].median())

# 小グループ作成の基準となる列を指定
print(df2.groupby("Survived").mean()["Age"])
print(df2.groupby("Pclass").mean()["Age"])

# ピボットテーブル機能を使う
print(pd.pivot_table(df2,index="Survived",columns="Pclass",values="Age"))

# 引数aggfuncを使って平均値以外の統計量を求める
print(pd.pivot_table(df2, index = "Survived", columns = "Pclass", values = "Age", aggfunc = max))

# Age列の欠損値の行を抜き出す（欠損であればTrue）
is_null = df2["Age"].isnull()

# Pclass1に関する埋め込み
df2.loc[(df["Pclass"] == 1) & (df["Survived"] == 0) & (is_null), "Age"] = 43
df2.loc[(df["Pclass"] == 1) & (df["Survived"] == 1) & (is_null), "Age"] = 35

# Pclass2に関する埋め込み
df2.loc[(df["Pclass"] == 2) & (df["Survived"] == 0) & (is_null), "Age"] = 33
df2.loc[(df["Pclass"] == 2) & (df["Survived"] == 1) & (is_null), "Age"] = 25

# Pclass3に関する埋め込み
df2.loc[(df["Pclass"] == 3) & (df["Survived"] == 0) & (is_null), "Age"] = 26
df2.loc[(df["Pclass"] == 3) & (df["Survived"] == 1) & (is_null), "Age"] = 20

# learn関数を使ってモデルに再学習させる
# 特徴量として利用する列のリスト
col = ["Pclass", "Age", "SibSp", "Parch","Fare"]
x = df2[col]
t = df2["Survived"]

for j in range(1,15):
    s1,s2,m = learn(x,t,depth= j)
    print(f"深さ{j}:訓練データの精度{s1}:テストデータの精度{s2}")

# groupbyメソッドを使って平均値を求める
sex = df2.groupby("Sex").mean()
print(sex["Survived"])

# # plotメソッドで棒グラフを作成する
# sex["Survived"].plot(kind = "bar")

# get_dummies関数で文字列を数値に変換する
male = pd.get_dummies(df2["Sex"], drop_first = True)
print(male)

# drop_firstを指定しないget_dummies関数の戻り値
print(pd.get_dummies(df["Sex"]))

# Embarked列をダミー変数化する
print(pd.get_dummies(df2["Embarked"],drop_first = True))

# drop_firstをFalseにしてみた場合
embarked = pd.get_dummies(df2["Embarked"],drop_first = False)
print(embarked.head())

# concat関数で2つのデータフレームを横方向に連結
x_temp = pd.concat([x,male], axis = 1)
print(x_temp.head())

# モデルの再学習
for j in range(1,6):
    s1,s2,m = learn(x_temp,t,depth=j)
    print(f"深さ{j}:訓練データの精度{s1}:テストデータの精度{s2}")

#　木の深さを５に指定して改めて学習
s1, s2, model = learn(x_temp,t,depth= 5)

import pickle
with open("survived.pkl", "wb") as f:
    pickle.dump(model,f)

# feature_importances_で特徴量重要度を確認
print(model.feature_importances_)

# データフレームに変換して表示
print(pd.DataFrame(model.feature_importances_, index = x_temp.columns))