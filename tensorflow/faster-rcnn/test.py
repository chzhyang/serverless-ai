
from six import BytesIO

# For drawing onto the image.
# import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import tempfile

def resize_image(url, new_width=256, new_height=256):
  _, filename = tempfile.mkstemp(suffix=".jpg")
#   image_data = response.read()
#   image_data = BytesIO(image_data)
  pil_image = Image.open(url)
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  print("Image resized")

  return filename


def test():
  print(resize_image("/tmp/input_0.jpg", 640, 480))

if __name__ == '__main__':

    # module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
    # detector = hub.load(module_handle).signatures['default']

    # image_urls =[
    #     "/faster-rcnn/data/input_0.jpg",
    #     "/faster-rcnn/data/input_1.jpg",
    # ]
    # print("input-0:\n")
    # detect_img(image_urls[0],detector)
    # print("input-1:\n")
    # detect_img(image_urls[1],detector)
    test()

