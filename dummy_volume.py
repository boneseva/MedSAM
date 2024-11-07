import os

import nibabel
import numpy as np


def crop_volume(volume, bounding_box):
    """Crop a 3D volume to a bounding box."""
    return volume[bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3], bounding_box[4]:bounding_box[5]]


def load_volume(volume_path):
    """Load a 3D volume from a NIfTI file."""
    volume = nibabel.load(volume_path)
    return volume.get_fdata()


def save_volume(volume, volume_path):
    """Save a 3D volume to a NIfTI file."""
    nibabel.save(nibabel.Nifti1Image(volume, np.eye(4)), volume_path)


if __name__ == "__main__":
    input = r"C:\Users\bonese\Documents\Courses\LRDL\LRDL-project\Self-supervised-Fewshot-Medical-Image-Segmentation\data\CHAOSCT\chaos_CT_normalized"
    output = r"C:\Users\bonese\Documents\Courses\LRDL\LRDL-project\Self-supervised-Fewshot-Medical-Image-Segmentation\data\CHAOSCT\cropped"

    if not os.path.exists(output):
        os.makedirs(output)

    input_image = os.path.join(input, "image_1.nii.gz")
    input_superpixel_large = os.path.join(input, "superpix-XLARGE_1.nii.gz")
    input_superpixel_small = os.path.join(input, "superpix-SMALL_1.nii.gz")
    input_superpixel_middle = os.path.join(input, "superpix-MIDDLE_1.nii.gz")

    bounding_box = [0, 25, 0, 25, 0, 10]
    save_volume(crop_volume(load_volume(input_image), bounding_box), os.path.join(output, "image_1-cropped.nii.gz"))
    save_volume(crop_volume(load_volume(input_superpixel_large), bounding_box), os.path.join(output, "superpix-XLARGE_1-cropped.nii.gz"))
    save_volume(crop_volume(load_volume(input_superpixel_small), bounding_box), os.path.join(output, "superpix-SMALL_1-cropped.nii.gz"))
    save_volume(crop_volume(load_volume(input_superpixel_middle), bounding_box), os.path.join(output, "superpix-MIDDLE_1-cropped.nii.gz"))
