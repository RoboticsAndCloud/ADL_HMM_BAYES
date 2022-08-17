# Author: Fei
# Date: 05/14/2021
# Brief: Compute the percentage of difference between 2 JPEG images of the same size
# Run: python3 diff_two_imgs.py 
# Reference: 1) https://pillow.readthedocs.io/en/stable/reference/Image.html
#            2): https://rosettacode.org/wiki/Percentage_difference_between_images#Python
#

#Import required Image library
from PIL import Image
from PIL import ImageChops

# path_one = 'test_original_resize_esp_224_224.jpg'
# path_two = 'test_original_pil_224_224.jpg'
prefix = 't1_original'
path_one = prefix + '_resize_esp_224_224.jpg'
path_two = prefix + '_resize_pil_224_224.jpg'

image_one = Image.open(path_one)
image_two = Image.open(path_two)

diff = ImageChops.difference(image_one, image_two)

if diff.getbbox():
    print("images are different")
    diff.show()
    diff.save(prefix + '_' + 'diff_btw_images.jpg')
else:
    print("images are the same")


# compare method https://rosettacode.org/wiki/Percentage_difference_between_images#Python
from PIL import Image

# path_one = 'test_original_resize_esp_224_224.jpg'
# path_two = 'test_original_pil_224_224.jpg'
i1 = Image.open(path_one)
i2 = Image.open(path_two)
assert i1.mode == i2.mode, "Different kinds of images."
assert i1.size == i2.size, "Different sizes."
 
pairs = zip(i1.getdata(), i2.getdata())
if len(i1.getbands()) == 1:
    # for gray-scale jpegs
    dif = sum(abs(p1-p2) for p1,p2 in pairs)
else:
    dif = sum(abs(c1-c2) for p1,p2 in pairs for c1,c2 in zip(p1,p2))
 
ncomponents = i1.size[0] * i1.size[1] * 3
print ("Difference (percentage):", (dif / 255.0 * 100) / ncomponents) 

# 1.614471309357076  test_original
# 0.5214924511471256  labframe
# 2.4101776127117516  t1_original


