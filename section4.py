import pandas as pd
data = {
    "松田の労働時間" : [160,160], #松田の労働時間列の作成

    "浅木の労働時間" :[161,175] #浅木の労働時間列の作成
    }

df = pd.DataFrame(data)

print(df)

print(type(df))
print(df.shape)

df.index = ["4月","5月"] # dfのインデックスを変更
print(df)
print(df.index)

df.columns = ["松田の労働(h)","浅木の労働(h)"] # 列名の変更
print(df)
print(df.columns)

data = [
    [160,161],
    [160,175]
    ]

df2 = pd.DataFrame(data,index = ["4月","5月"], columns = ["松田の労働(h)","浅木の労働(h)"])

print(df2)

# pandasは別名pdでインポート済
# KvsT.csvファイルを読込んでデータフレームに変換
df = pd.read_csv("KvsT.csv")
print(df.head())
print(df.tail())

print(df["身長"])

col = ["身長","体重"]

print(df[col])

print(type(df["派閥"]))
print(type(df[col]))

print(df["派閥"])

# 特徴量の列を参照してｘに代入
xcol = ["身長","体重","年代"]
x = df[xcol]
print(x)

# 正解データ（派閥）を参照して、tに代入
t = df["派閥"]
print(t)

from sklearn import tree

# モデルの学習（未学習）
model = tree.DecisionTreeClassifier(random_state=0)

# 学習の実行
model.fit(x,t)

# TAROのデータを2次元リストで作成
taro = [[170,70,20]]

# TAROがどちらに分類されるか予測
print(model.predict(taro))

#複数の予測を一度に実行
matsuda = [172,65,20] # 松田のデータ
asagi = [158,48,20] #浅木のデータ
new_data = [matsuda,asagi] #2人のデータを二次元リスト化
print(model.predict(new_data)) #2人のデータを一括で予測

# 正答率の計算
print(model.score(x,t))

# モデルの保存
import pickle
with open("KinokoTakenoko.pkl", "wb") as f:
    pickle.dump(model,f)

# KinokoTakenoko.pklからモデルを変数に読込む
import pickle

with open("KinokoTakenoko.pkl","rb") as f:
    model2 = pickle.load(f)

# ファイルから読込んだ学習済モデルで予測する
suzuki = [[180,75,39]]
print(model2.predict(suzuki))