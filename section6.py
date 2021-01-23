import pandas as pd

df = pd.read_csv("cinema.csv")
print(df.head())

# 欠損値があるか確認
print(df.isnull().any(axis=0))

# 欠損値を平均値で保管してdf2に代入
df2 = df.fillna(df.mean())
print(df2.isnull().any(axis=0))

# # グラフ描画するためのおまじない(Notebookで実行）
# %matplotlib inline
#
# # SNS2とsalesの散布図の作成
# df2.plot(kind = "scatter",x = "SNS1", y = "sales")
# df2.plot(kind = "scatter",x = "SNS2", y = "sales")
# df2.plot(kind = "scatter",x = "actor", y = "sales")
# df2.plot(kind = "scatter",x = "original", y = "sales")

# (補足)データフレームを作成して特定の行を参照する
# データフレームの作成
test = pd.DataFrame({"Acolumns": [1, 2, 3], "Bcolumns": [4, 5, 6]})

# Acolumns列の値が2未満の行だけ参照する
print(test[test["Acolumns"] < 2])

# Acolumns列に対して比較演算を行う
print(test["Acolumns"] < 2)

# dropメソッドでインデックスが０の行を削除する
print(test.drop(0, axis=0))

# 列を削除する
print(test.drop("Bcolumns", axis=1))

# 2つの条件で外れ値の行を特定する
print(df[(df["SNS2"] > 1000) & (df["sales"] < 8500)])

# 特定した行からインデックスのみを取り出す
no = df2[(df["SNS2"] > 1000) & (df["sales"] < 8500)].index
print(no)

# 外れ値を含む行の削除
df3 = df2.drop(30, axis=0)
print(df3.shape)

# df3から特徴量の変数xと正解データの変数tに分割
col = ["SNS1","SNS2","actor","original"]
x = df3[col] #特徴量の取り出し
t = df3["sales"] # 正解データの取り出し

# インデックス２の行からSNS1列の値を取り出す
print(df3.loc[2,"SNS1"])

# 特定のデータのみを参照する
index = [2,4,6] # インデックス
col = ["SNS1", "actor"] #列名
print(df3.loc[index,col])

# スライス構文で連続した要素を参照する
sample = [10,20,30,40]
print(sample[1:3])

# データフレームで複数のインデックスや列名を参照する
# 0行目以上2行目以下、actor列より左の列（actor列を含む）
print(df3.loc[0:3, :"actor"])

# スライス構文で特徴量と正解データを取り出す
x = df3.loc[ : ,"SNS1":"original"] #特徴量の取り出し
t = df3["sales"] # 正解ラベルの取り出し

# 訓練データとテストデータに分割する
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,t,test_size = 0.2, random_state = 0)

# 重回帰モデルのLinearRegression関数をインポートする
from  sklearn.linear_model import LinearRegression

# LinearRegression関数を使ってモデルを作成する
model = LinearRegression()
model.fit(x_train,y_train)

new = [[150,700,300,0]]
print(model.predict(new))

# MAEを求める
# 関数のインポート
from sklearn.metrics import mean_absolute_error

pred = model.predict(x_test)

# 平均絶対誤差の計算
print(mean_absolute_error(y_pred = pred,y_true = y_test))

# モデルを保存する
import pickle

with open("cinema.pkl","wb") as f:
    pickle.dump(model,f)

# scoreメソッド
print(model.score(x_test,y_test))

# 係数を切片を確認
print(model.coef_) #計算式の係数の表示
print(model.intercept_) #計算式の切片の表示

# 列と係数を表示する
tmp = pd.DataFrame(model.coef_) #データフレームの作成
tmp.index = x_train.columns #列名をインデックスに指定
print(tmp)

