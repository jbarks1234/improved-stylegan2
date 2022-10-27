from PIL import Image
import os, sys

path = "flower"
dirs = os.listdir(path)

def resize():
    for item in dirs:
        if os.path.isfile(path+"/"+item):
            im = Image.open(path+"/"+item)
            f, e = os.path.splitext(path+"/"+item)
            imResize = im.resize((512,512), Image.ANTIALIAS)
            imResize.save("./lucid_stylegan/flowers2" + "/"+item, 'JPEG', quality=90)

resize()