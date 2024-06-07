import os
import csv
import json


# 画像が保存されているディレクトリをロードする
images = os.listdir("")
images = sorted(images)

# プロンプトを設定する
prompt = ""


# case: jsonl
data = []
for i in images:
    data.extend([{"file_name": i, "text": prompt}])

with open("metadata.jsonl", "w") as file:
    for entry in data:
        json_str = json.dumps(entry)  # オブジェクトをJSON文字列に変換
        file.write(json_str + "\n")  # 1行に1つのJSONオブジェクトを保存

# case: csv
data = [["file_name","text"]]
for i in images:
    data.extend([[i, prompt]])

with open("metadata.csv", "w", newline='') as f:
    writer = csv.writer(f)
    for row in data:
        writer.writerow(row)