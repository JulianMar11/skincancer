import os
'''
# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk("./data/CorrosionImages"):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        print(path[1])
        print(path[2])
        mypath = os.path.join(*path)
        filepath= os.path.join(mypath, file)
        print(filepath)
        print(len(path) * '---', file)

'''

import PATH as Path
import models.ImageData as ImgData

shape = (200, 200, 1)
paths = Path.paths_local
data = ImgData.Data(paths[2], shape, classes=3)
data.load()
x, y = data.random_batch(50, mode="train")
print(x.shape)
print(y.shape)

ImgData.show_batch(x, y)
# x, y = data.random_batch(20, mode="test")
# ImgData.show_batch(x, y)
