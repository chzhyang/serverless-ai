# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides utilities to preprocess images.

Images used during evaluation are resized (with aspect-ratio preservation) and
centrally cropped. All images undergo mean color subtraction.

"""

import tensorflow as tf

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
_RESIZE_MIN = 256

def _central_crop(image, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image: a 3-D image tensor
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    3-D tensor with cropped image.
  """
  shape = tf.shape(input=image)
  height, width = shape[0], shape[1]

  amount_to_be_cropped_h = (height - crop_height)
  crop_top = amount_to_be_cropped_h // 2
  amount_to_be_cropped_w = (width - crop_width)
  crop_left = amount_to_be_cropped_w // 2
  return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def _mean_image_subtraction(image, means, num_channels):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
    num_channels: number of color channels in the image that will be distorted.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')

  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  means = tf.broadcast_to(means, tf.shape(image))

  return image - means


def _smallest_size_at_least(height, width, resize_min):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    resize_min: A python integer or scalar `Tensor` indicating the size of the
      smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: an int32 scalar tensor indicating the new width.
  """
  resize_min = tf.cast(resize_min, tf.float32)

  # Convert to floats to make subsequent calculations go smoothly.
  height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

  smaller_dim = tf.minimum(height, width)
  scale_ratio = resize_min / smaller_dim

  # Convert back to ints to make heights and widths that TF ops will accept.
  new_height = tf.cast(height * scale_ratio, tf.int32)
  new_width = tf.cast(width * scale_ratio, tf.int32)

  return new_height, new_width


def _aspect_preserving_resize(image, resize_min):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    resize_min: A python integer or scalar `Tensor` indicating the size of the
      smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  shape = tf.shape(input=image)
  height, width = shape[0], shape[1]

  new_height, new_width = _smallest_size_at_least(height, width, resize_min)

  return _resize_image(image, new_height, new_width)


def _resize_image(image, height, width):
  """Simple wrapper around tf.resize_images.

  This is primarily to make sure we use the same `ResizeMethod` and other
  details each time.

  Args:
    image: A 3-D image `Tensor`.
    height: The target height for the resized image.
    width: The target width for the resized image.

  Returns:
    resized_image: A 3-D tensor containing the resized image. The first two
      dimensions have the shape [height, width].
  """
  return tf.compat.v1.image.resize(
      image, [height, width],
      method=tf.image.ResizeMethod.BILINEAR,
      align_corners=False)


def preprocess_image(image_buffer, output_height, output_width, num_channels):
  """Preprocesses the given image.

  Preprocessing includes decoding, cropping, and resizing images.

  Args:
    image_buffer: scalar string Tensor representing the raw JPEG image buffer.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    num_channels: Integer depth of the image buffer for decoding.

  Returns:
    A preprocessed image.
  """
  image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
  image = _aspect_preserving_resize(image, _RESIZE_MIN)
  image = _central_crop(image, output_height, output_width)

  image.set_shape([output_height, output_width, num_channels])

  return _mean_image_subtraction(image, CHANNEL_MEANS, num_channels)