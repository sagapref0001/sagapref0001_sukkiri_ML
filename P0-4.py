# 「10、9、8、・・・ 2、1、Lift Off!」のようなカウントダウンを行う
for i in range(10,0,-1):
    print(f"{i}、",end = "")
print("Lift off!")

# 試験得点の平均点を計算して表示

# scoreリストを作成
scores = []

# for文を利用して表のデータをscoreリストに格納
for i in range(10):
    score = int(input(f"{i+1}人目の点数"))
    scores.append(score)

# scoreリストの各要素を加工してfinal_listに格納
final_scores = []

for score in scores:
    final_scores.append(0.8 * score + 20)

# final_listの平均点を計算して表示
print(f"平均点は{sum(final_scores) / len(final_scores)}点")
