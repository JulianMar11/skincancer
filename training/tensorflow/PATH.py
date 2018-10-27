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

paths_remote = {"database": "/home/Julian/DATA/",
                "train_results": "/home/Julian/BeeNets/data/results/",
                "pollen": "/home/Julian/DATA/Pollen_Dataset/",
                "cifar10": "/home/Julian/DATA/cifar-10-batches-py/",
                "yolo": "/home/Julian/DATA/abgabe_trained_ep010-loss31.293-val_loss31.551.h5"}

test_videos_local = ["/Users/Julian/Desktop/Dropbox/synthbeedata/videos/31_05_2018_Hopfner_video_10.mp4",
                     "/Users/Julian/Desktop/Dropbox/synthbeedata/videos/31_05_2018_Hopfner_video_100.mp4",
                     "/Users/Julian/Desktop/Dropbox/synthbeedata/videos/31_05_2018_Hopfner_video_970.mp4"]

test_videos_remote = ["/home/ubuntu/DATA/videos/31_05_2018_Hopfner_video_10.mp4",
                      "/home/ubuntu/DATA/videos/31_05_2018_Hopfner_video_100.mp4",
                      "/home/ubuntu/DATA/videos/31_05_2018_Hopfner_video_970.mp4"]

# Paths to data depending of USER variable
paths = paths_local \
    if os.getenv('HOME') == '/Users/Julian' \
    else paths_production

videos = test_videos_local \
    if os.getenv('USER') == 'Julian' \
    else test_videos_remote
