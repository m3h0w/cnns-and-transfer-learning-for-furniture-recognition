from PIL import Image
import os, sys
import ntpath

path = "./data/train/"
output_path = './data/train/244/'
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((244,244), Image.ANTIALIAS)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            print(f)
            print(ntpath.basename(f))
            imResize.save(output_path + ntpath.basename(f) + '.jpg', 'JPEG', quality=90)

resize()