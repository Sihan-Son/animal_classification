# Created by sihan at 2018-11-03

from classification.AnimalClassification.classification import AnimalClassification
import time

now = time.time()
path = r'../test_file/1.jpg'
test = AnimalClassification(path)
print(test.run_graph())
print(time.time() - now)
