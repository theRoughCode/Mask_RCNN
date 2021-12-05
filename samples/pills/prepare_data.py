import tensorflow as tf
import cv2
import random

def get_mask(img):
  # Get mask
  mask = cv2.inRange(img, np.array([114, 114, 114]), np.array([120, 120, 120]))
  mask = cv2.bitwise_not(mask)

  # Remove noise
  kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(10, 10))
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

  # Flood fill
  h, w = mask.shape[:2]
  zero_mask = np.zeros((h + 2, w + 2), np.uint8)

  # Get top left
  top_left_mask = mask.copy()
  cv2.floodFill(top_left_mask, zero_mask, (0, 0), 255,)

  # Get top right
  top_right_mask = mask.copy()
  cv2.floodFill(top_right_mask, zero_mask, (w - 1, 0), 255)
  top_mask = cv2.bitwise_xor(top_left_mask, top_right_mask)

  # Get bottom left
  bottom_left_mask = mask.copy()
  cv2.floodFill(bottom_left_mask, zero_mask, (0, h - 1), 255)

  # Get bottom right
  bottom_right_mask = mask.copy()
  cv2.floodFill(bottom_right_mask, zero_mask, (w - 1, h - 1), 255)
  bottom_mask = cv2.bitwise_xor(bottom_left_mask, bottom_right_mask)

  # Merge masks
  mask = cv2.bitwise_or(top_mask, bottom_mask)
  mask = cv2.bitwise_not(mask)

  return mask

def resize_img(img, mask):
  MIN_DIM, MAX_DIM = 100, 400
  h, w = img.shape[:2]

  larger_dim = max(h, w)
  max_scale = MAX_DIM / larger_dim
  min_scale = MIN_DIM / larger_dim
  scale = random.random() * (max_scale - min_scale) + min_scale

  # Resize image
  h, w = int(h * scale), int(w * scale)
  img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
  mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

  return img, mask

def rotate_img(img, mask):
  MAX_ANGLE = 360
  h, w = img.shape[:2]
  centre = (w / 2, h / 2)

  while True:
    angle = random.random() * MAX_ANGLE
    rot_mat = cv2.getRotationMatrix2D(centre, angle, 1.0)
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)
    if bound_w <= 580 and bound_h <= 580:
      break

  # subtract old image center (bringing image back to origin) and add the new image center coordinates
  rot_mat[0, 2] += bound_w/2 - centre[0]
  rot_mat[1, 2] += bound_h/2 - centre[1]

  # rotate image with the new bounds and translated rotation matrix
  rotated_img = cv2.warpAffine(img, rot_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR)
  rotated_mask = cv2.warpAffine(mask, rot_mat, (bound_w, bound_h), flags=cv2.INTER_NEAREST)

  return rotated_img, rotated_mask

def augment_img(pill):
  mask = get_mask(pill)
  pill, mask = resize_img(pill, mask)
  pill, mask = rotate_img(pill, mask)
  return pill, mask

def augment_bg(bg, out_dim):
  # Random zoom in
  MAX_SCALE = 1.5
  scale = random.random() * (MAX_SCALE - 1.) + 1.
  bg = cv2.resize(bg, (int(bg.shape[1] * scale), int(bg.shape[0] * scale)), interpolation=cv2.INTER_AREA)

  # Random flip
  flip = random.randint(-1, 2)
  # If flip == 2: do nothing
  if flip != 2:
    bg = cv2.flip(bg, flip)

  # Crop
  min_dim = min(bg.shape[0], bg.shape[1])
  if min_dim < out_dim:
    bg = cv2.resize(bg[:min_dim, :min_dim], (out_dim, out_dim))
  else:
    # Get random crop
    x1 = random.randint(0, bg.shape[1] - out_dim)
    y1 = random.randint(0, bg.shape[0] - out_dim)
    x2, y2 = x1 + out_dim, y1 + out_dim
    bg = bg[y1:y2, x1:x2]

  return bg


def overlay_images(pill, bg, bg_dim=600, padding=10):
  """
  Overlays pill onto background with augmentations.
  """
  # Augment pill and get mask
  pill, mask = augment_img(pill)
  
  h, w = pill.shape[:2]
  bin_mask = mask / 255.0
  bin_mask = np.repeat(bin_mask[:, :, np.newaxis], 3, axis=2)

  # Get random location
  min_x, max_x = padding, bg_dim - w - padding
  min_y, max_y = padding, bg_dim - h - padding
  x1, y1 = random.randint(min_x, max_x), random.randint(min_y, max_y)
  x2, y2 = x1 + w, y1 + h
  
  # Augment background
  out = augment_bg(bg, bg_dim)

  # Create shadow(s)
  num_shadows = random.randint(0, 3)
  for _ in range(num_shadows):
    # Generate shadow offset from pill
    MAX_OFFSET_X, MAX_OFFSET_Y = w // 10, h // 10
    offset_x = random.randint(-min(MAX_OFFSET_X, x1), min(MAX_OFFSET_X, max_x - x1))
    offset_y = random.randint(-min(MAX_OFFSET_Y, y1), min(MAX_OFFSET_Y, max_y - y1))
    shadow_x1, shadow_x2 = x1 + offset_x, x2 + offset_x
    shadow_y1, shadow_y2 = y1 + offset_y, y2 + offset_y
    # Generate blurred shadow
    opacity = random.random()
    blur_size = random.randint(2, 30) * 2 + 1 # Blur kernel = 5, 7, 9, ..., 59, 61
    shadow_mask = cv2.GaussianBlur(bin_mask * opacity, (blur_size, blur_size), 0)
    # Overlay shadow on bg
    shadow = np.zeros_like(bin_mask)
    out[shadow_y1:shadow_y2, shadow_x1:shadow_x2] = (1.0 - shadow_mask) * out[shadow_y1:shadow_y2, shadow_x1:shadow_x2] + shadow_mask * shadow

  # Overlay pill on bg
  out[y1:y2, x1:x2] = (1.0 - bin_mask) * out[y1:y2, x1:x2] + bin_mask * pill
  
  # Create mask
  mask = np.zeros_like(out)
  mask[y1:y2, x1:x2] = bin_mask

  # Apply affine transformation
  MAX_DEVIATION = padding // 2
  deviation = np.float32(np.random.randint(-MAX_DEVIATION, MAX_DEVIATION, size=(3, 2)))
  pts1 = np.float32([[padding,padding],[200,padding],[padding,200]])
  pts2 = pts1 + deviation
  M = cv2.getAffineTransform(pts1, pts2)
  out = cv2.warpAffine(out, M, (bg_dim, bg_dim))
  mask = cv2.warpAffine(mask, M, (bg_dim, bg_dim))

  # Add random brightness/contrast
  MAX_BRIGHTNESS = 50
  brightness = random.randint(-MAX_BRIGHTNESS, MAX_BRIGHTNESS)
  MAX_CONTRAST_DELTA = 0.2
  contrast = random.random() * (MAX_CONTRAST_DELTA * 2) + (1.0 - MAX_CONTRAST_DELTA)
  out = cv2.convertScaleAbs(out, alpha=contrast, beta=brightness)

  # Add gaussian blur
  blur_size = random.randint(0, 6) * 2 + 1  # blur size = 1, 3, ..., 11, 13
  out = cv2.GaussianBlur(out, (blur_size, blur_size), 0)

  return out, mask
