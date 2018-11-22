# Create by sihan
# Date : 2018-11-22

from classification.AnimalClassification.classification import AnimalClassification

path = r'../test_file/1.jpg'

test = AnimalClassification(path)

# 이미지 결과 라벨과 [고양이 확률, 강아지 확률] 반환
label, prob = test.run_graph()

print(label)
print(prob)
