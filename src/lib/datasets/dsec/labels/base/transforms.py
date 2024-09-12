import torch
import numpy as np


class ToTensor:
    def __call__(self, sample):
        sample = torch.from_numpy(sample)

        return sample


class Padding:
    def __init__(self, img_height, img_width, no_labels_value):
        self.img_height = img_height
        self.img_width = img_width
        self.no_labels_value = no_labels_value

    def __call__(self, sample):
        ori_height, ori_width = sample.shape[:2]
        top_pad = self.img_height - ori_height
        right_pad = self.img_width - ori_width

        assert top_pad >= 0 and right_pad >= 0

        sample[0] = np.lib.pad(sample[0],
                            ((top_pad, 0), (0, right_pad)),
                            mode='constant',
                            constant_values=self.no_labels_value)

        return sample

# outputs = [left_img, calib, calib_R, image_index, target]

class Crop:
    def __init__(self, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(self, sample, offset_x, offset_y):

        start_y, end_y = offset_y, offset_y + self.crop_height
        start_x, end_x = offset_x, offset_x + self.crop_width

        # left_img crop
        sample[0] = sample[0][start_y:end_y, start_x:end_x]


        return sample




class VerticalFlip:
    def __call__(self, sample):
        sample[0] = np.copy(np.flipud(sample[0]))
        sample[4] = sample[4].transpose(0)

        return sample
