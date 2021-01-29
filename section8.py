import pandas as pd
# %matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("Boston.csv")
print(df.head())

# CRIME列にデータが何種類あるか調べる
print(df["CRIME"].value_counts())

# ダミー変数化した列を連結しCRIME列を削除
crime = pd.get_dummies(df["CRIME"],drop_first = True)
df2 = pd.concat([df,crime],axis = 1)
df2 = df2.drop(["CRIME"],axis = 1)
print(df2.head())

# 訓練データ＆検証データとテストデータに分類する
train_val,test = train_test_split(df2,test_size = 0.2, random_state = 0)

# train_valの欠損値を確認する
print(train_val.isnull().sum())

# 欠損値を平均値で穴埋めする
train_val_mean = train_val.mean() #各列の平均値の計算
train_val2 = train_val.fillna(train_val_mean) #平均値で穴埋め

# # 各特徴量の列とPRICE列の相関関係を示す散布図を描く
# colname = train_val2.columns
# for name in colname:
#     train_val2.plot(kind = "scatter", x = name, y = "PRICE")

# 外れ値が存在するインデックスを確認する
# RM列の外れ値
out_line1 = train_val2[(train_val2["RM"] < 6) &
                       (train_val2["PRICE"] > 40)].index
out_line2 = train_val2[(train_val2["PTRATIO"] > 18) &
                       (train_val2["PRICE"] > 40)].index
print(out_line1,out_line2)

# 外れ値を削除する
train_val3 = train_val2.drop([76], axis = 0)

# 絞り込んだ列以外を取り除く
col = ["INDUS", "NOX", "RM", "PTRATIO", "LSTAT","PRICE"]

train_val4 = train_val3[col]
print(train_val4.head())

# 列同士の相関関係を調べる
print(train_val4.corr()["PRICE"])

# 各列とPRICE列との相関関係を見る
train_cor = train_val4.corr()["PRICE"]

#mapメソッドで要素に関数を適用する
se = pd.Series([1,-2,3,-4]) #シリーズの作成

# seの各要素にABS関数を適応させた結果をシリーズ化
se.map(abs)

# 相関行列のPRICE列と相関係数を絶対値に変換する
abs_cor = train_cor.map(abs)
print(abs_cor)

# sort_valueメソッドで要素を降順に並べ替える
print(abs_cor.sort_values(ascending = False))

# 訓練データと検証データに分類する
col = ["RM","LSTAT","PTRATIO"]
x = train_val4[col]
t = train_val4[["PRICE"]]
x_train, x_val, y_train, y_val = train_test_split(x,t,test_size = 0.2,random_state = 0)

# scikit-learnのpreprocessingモジュールを使う
from sklearn.preprocessing import StandardScaler
sc_model_x = StandardScaler()
sc_model_x.fit(x_train)

# 各列のデータを標準化してsc_xに代入
sc_x = sc_model_x.transform(x_train)
print(sc_x)

# # 平均値０を確認する
# # array型だと見づらいのでデータフレームに変換
tmp_df = pd.DataFrame(sc_x,columns = x_train.columns)

# 平均値の計算
print(tmp_df.mean())

# 標準偏差１を確認する
print(tmp_df.std())

# 正解データを標準化する
sc_model_y = StandardScaler()
sc_model_y.fit(y_train)

sc_y = sc_model_y.transform(y_train)

# 標準化したデータで学習させる
model = LinearRegression()
model.fit(sc_x,sc_y) #標準化済みの訓練データで学習

# scoreメソッドで決定係数を求める
print(model.score(x_val,y_val))

# 検証データを標準化する
sc_x_val = sc_model_x.transform(x_val)
sc_y_val = sc_model_y.transform(y_val)
# 標準化した検証データを決定係数を計算
print(model.score(sc_x_val, sc_y_val))

# learn関数の定義
def learn(x,t):
    x_train, x_val, y_train, y_val = train_test_split(x,t,test_size= 0.2, random_state= 0)
    # 訓練データを標準化
    sc_model_x = StandardScaler()
    sc_model_y = StandardScaler()
    sc_molel_x.fit(x_train)
    sc_x_train = sc_model_x.transform(x_train)
    sc_model_y.fit(y_train)
    sc_y_train = sc_model.y.transform(y_train)
    # 学習
    model = LinearRegression()
    mpdel.fit(sc_X_train, sc_y_train)
    # 検証データを標準化
    sc_x_val = sc_model_x.transform(x_val)
    sc_y_val = sc_model_y.transform(y_val)
    # 訓練データと検証データの決定係数計算
    train_score = model.score(sc_x_train, sc_y_train)
    val_score = model.score(sc_x_val,sc_y_val)
    return train_score,val_score
x
