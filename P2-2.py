# 3科目の試験得点を管理するリストを１つ作成
test_score = []

# 国語の試験得点をキーボートから入力
japanese_score = int(input("国語の試験得点"))

# 国語の試験得点を手順１で作成したリストに追加
test_score.append(japanese_score)

# 数学の試験得点をキーボードから入力
math_score = int(input("数学の試験得点"))

# 数学の得点を手順１で作成したリストに追加
test_score.append(math_score)

# 英語の試験得点をキーボードから入力
english_score = int(input("数学の試験得点"))

# 英語の得点を手順１で作成したリストに追加
test_score.append(english_score)

# リストの一覧を表示
print(test_score)

# リストの合計値を計算して表示
print(sum(test_score))

