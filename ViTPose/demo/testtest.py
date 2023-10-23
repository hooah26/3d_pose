import os

folder_path = './Sample_test_video'
file_list = os.listdir(folder_path)
video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
video_files = []

for file_name in file_list:
    file_extension = os.path.splitext(file_name)[-1].lower()
    if file_extension in video_extensions:
        video_path = os.path.join(folder_path, file_name)
        video_files.append(video_path)

for video_file in video_files:
    print("video_file!!!", video_file)
    print(len(video_files))

