# Created by sihan at 2018-11-03

from .packing import AnimalClassification

path = r'D:\Project\1.jpg'
test = AnimalClassification(path)
print(test.run_graph())
