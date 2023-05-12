import gdown
import shutil
import os


gdown.download(
    'https://drive.google.com/file/d/1RjFrBLyNTjzYN9ow8HVfQFl3CYcGz8qM/view?usp=sharing', 
    quiet=False,fuzzy=True)
shutil.unpack_archive('dataset.zip', '')
os.remove('dataset.zip')
