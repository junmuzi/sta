import os

path="/media/lijun_private_datasets/data_30/hmdb51"

num = {}

for class_name in os.listdir(path):
    class_path = os.path.join(path, class_name)
    for video_name in os.listdir(class_path):
        video_path = os.path.join(class_path, video_name)
        n_frame = len(os.listdir(video_path)) - 1
        if not n_frame in num:
            num[n_frame] = 0
        num[n_frame] = 1
        print(video_path)
print(num)
