# Created by sihan at 2018-11-03

import time

from classification.AnimalClassification.classification import AnimalClassification

now = time.time()
path = r'../test_file/1.jpg'
test = AnimalClassification(path)

a, b = test.run_graph()

print(a)
print(b)
print(time.time() - now)
