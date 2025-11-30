import glob
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import StandardScaler
from image_loader import ImageLoader

def compute_mean_and_std(dir_name: str) -> (np.array, np.array):
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then in [0,1] before computing mean
  and standard deviation

  Tip: You can use any function you want to find mean and ssttd deviation

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None

  ############################################################################
  # Student code begin
  ############################################################################
  # Load data
  all_paths = glob.glob(dir_name + "**/**/*.jpg", recursive=True)

  # Allows us to compute online mean and std
  scaler = StandardScaler()

  for img_path in all_paths:
    # Grayscale
    cur_image = Image.open(img_path).convert("L")

    # 0,1
    cur_image = np.asarray(cur_image, dtype=np.float64) / 255.0

    # partial_fit expects (n_samples, n_features) assignment says n_features = 1
    cur_image = cur_image.reshape(-1, 1)

    scaler.partial_fit(cur_image)

  mean = scaler.mean_
  std = np.sqrt(scaler.var_)
  ############################################################################
  # Student code end
  ############################################################################
  return mean, std
