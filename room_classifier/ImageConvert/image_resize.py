# Author: Fei
# Date: 05/14/2021
# Brief: Resize the image using PIL
# Reference: 1) https://pillow.readthedocs.io/en/stable/reference/Image.html
#            2) Pil image resize source code: https://pillow.readthedocs.io/en/stable/_modules/PIL/Image.html#Image.resize

#Import required Image library
from PIL import Image

import timeit
import time

from timeit import default_timer as timer

# print("Start : %s" % time.ctime())
start = timer()

#Create an Image Object from an Image
image_file = 'test_original_240_176_generated_640' #test_original_240_176_generated_640.jpg
im = Image.open(image_file + '.jpg')
# print(im)  # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x480 at 0x7F80E662F5C0>

#Display actual image
im.show()

#Make the new image with new size: the width and half the height of the original image
resized_im = im.resize((224, 224), Image.BILINEAR)
end = timer()
print("Resizing the image takes:")
print(end - start)  
# 0.060595470014959574 t1_original
# print("End : %s" % time.ctime())


#Display the resized imaged
resized_im.show()

#Save the cropped image
resized_im.save(image_file + '_' + 'resize_pil_224_224' + '.jpg')
