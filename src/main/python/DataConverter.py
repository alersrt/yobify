import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras

dimen = 32

dir_path = os.path.abspath(__file__ + '/../../resources/image_data/') + '/'
output_path = os.path.abspath(__file__ + '/../../resources/processed_data/') + '/'

sub_dir_list = os.listdir(dir_path)
images = list()
labels = list()
fAmount = 0
for i in range(len(sub_dir_list)):
  if not os.path.isfile(sub_dir_list[i]):
    label = i
    image_names = os.listdir(dir_path + sub_dir_list[i])
    for image_path in image_names:
      path = dir_path + sub_dir_list[i] + '/' + image_path
      image = Image.open(path).convert('L')
      resize_image = image.resize((dimen, dimen))
      array = list()
      for x in range(dimen):
        sub_array = list()
        for y in range(dimen):
          sub_array.append(resize_image.load()[x, y])
        array.append(sub_array)
      image_data = np.array(array)
      image = np.array(np.reshape(image_data, (dimen, dimen, 1))) / 255
      images.append(image)
      labels.append(label)
    print(f'#{i} - [{sub_dir_list[i]}]')
  else:
    fAmount = fAmount + 1

x = np.array(images)
y = np.array(keras.utils.to_categorical(np.array(labels), num_classes=len(sub_dir_list) - fAmount))

train_features, test_features, train_labels, test_labels = train_test_split(x, y)

np.save('{}x.npy'.format(output_path), train_features)
np.save('{}y.npy'.format(output_path), train_labels)
np.save('{}test_x.npy'.format(output_path), test_features)
np.save('{}test_y.npy'.format(output_path), test_labels)
