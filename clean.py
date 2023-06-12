import os
import shutil
def clean_folders_with_few_files(path, min_files=2):
    for foldername, subfolders, filenames in os.walk(path, topdown=False):
        if len(filenames) < min_files:
            shutil.rmtree(foldername)
            print(f"Removed folder with few files: {foldername}")

# 用法示例
clean_folders_with_few_files("./ckpt", min_files=2)