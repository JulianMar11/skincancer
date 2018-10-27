import os

paths_local = {"dataset": "/Users/Julian/GitHub/skincancer/data/dataset",
               "train_results": "/Users/Julian/GitHub/skincancer/data/results/",
               "annotation_results": "/Users/Julian/GitHub/BeeNets/data/annotations/",
               "pollen": "/Users/Julian/Desktop/Dropbox/synthbeedata/Pollen_Dataset/",
               "cifar10": "/Users/Julian/Desktop/Dropbox/synthbeedata/cifar-10-batches-py/",
               "yolo": {
                   "model": "/Users/Julian/GitHub/BeeNets/models/yolo/logs/abgabe_trained_ep010-loss31.293-val_loss31.551.h5",
                   "classnames": "/Users/Julian/GitHub/BeeNets/models/yolo/input/classnames_bees.txt",
                   "anchors": "/Users/Julian/GitHub/BeeNets/models/yolo/input/anchorstrain.txt"
               }
               }

paths_production= {"dataset": "/Users/Matze/Desktop/HackathonScionWinter/data/dataset",
               "train_results": "/Users/Matze/Desktop/HackathonScionWinter/data/results/",
               "annotation_results": "/Users/Julian/GitHub/BeeNets/data/annotations/",
               "pollen": "/Users/Julian/Desktop/Dropbox/synthbeedata/Pollen_Dataset/",
               "cifar10": "/Users/Julian/Desktop/Dropbox/synthbeedata/cifar-10-batches-py/",
               "yolo": {
                   "model": "/Users/Julian/GitHub/BeeNets/models/yolo/logs/abgabe_trained_ep010-loss31.293-val_loss31.551.h5",
                   "classnames": "/Users/Julian/GitHub/BeeNets/models/yolo/input/classnames_bees.txt",
                   "anchors": "/Users/Julian/GitHub/BeeNets/models/yolo/input/anchorstrain.txt"
               }
               }

paths_pc = {"dataset2018": "/home/matthias/Schreibtisch/skincancer/data/dataset_2018",
            "dataset": "/home/matthias/Schreibtisch/skincancer/data/dataset",
            "train_results": "/home/matthias/Schreibtisch/skincancer/data/results/",
               }

paths_remote = {"database": "/home/Julian/DATA/",
                "train_results": "/home/Julian/BeeNets/data/results/",
                "pollen": "/home/Julian/DATA/Pollen_Dataset/",
                "cifar10": "/home/Julian/DATA/cifar-10-batches-py/",
                "yolo": "/home/Julian/DATA/abgabe_trained_ep010-loss31.293-val_loss31.551.h5"}

print(os.getenv('HOME'))
if os.getenv('HOME') == '/home/matthias':
    paths = paths_pc
elif os.getenv('HOME') == '/Users/Julian':
    paths = paths_local
elif os.getenv('HOME') == '/home/matze':
    paths = paths_production
else:
    paths=paths_remote
print(paths)