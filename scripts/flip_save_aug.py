# from PIL import Image, ImageOps

# img = Image.open('E:/projects/head_location/Implementation/CNN-based/dataset/train/head_left/img_1.png')
# img_flip = ImageOps.mirror(img)
# img_flip.save('E:/projects/head_location/Implementation/CNN-based/temp/flip/img_1_flip.png')

from PIL import Image, ImageOps
import os 
import glob

src = 'RIGHT'
dst = 'LEFT'

PATH_src = r"E:/projects/head_location/Implementation/CNN-based/dataset/train/head_{}".format(src)
PATH_dst = r"E:/projects/head_location/Implementation/CNN-based/dataset/train/head_{}".format(dst)
i = 1

for image in os.listdir(PATH_src):
	img_path = os.path.join(PATH_src, image)
	img = Image.open(img_path)
	img_flip = ImageOps.mirror(img)
	img_flip.save(PATH_dst + '/' + f'img_flip_{i}.png')
	i+=1
