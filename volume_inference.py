import uuid

import cv2
import nibabel as nib
import numpy as np
import subprocess
import tempfile
import os

import torch
from PIL import Image
from matplotlib import pyplot as plt

import MedSAM_Inference
from segment_anything import sam_model_registry
from skimage import io, transform

# Directory for storing intermediate files during debugging
DEBUG_TEMP_DIR = "C:/Users/bonese/Documents/Courses/LRDL/LRDL-project/MedSAM/temp_files"
os.makedirs(DEBUG_TEMP_DIR, exist_ok=True)


def normalize_to_uint8(slice_array):
    """Normalize a float array to uint8 (0-255) for PNG compatibility."""
    slice_min, slice_max = slice_array.min(), slice_array.max()
    normalized_slice = (slice_array - slice_min) / (slice_max - slice_min) * 255
    return normalized_slice.astype(np.uint8)


def segment_slice_with_sam(input_array, bounding_box=None):
    """Run SAM on a single normalized slice"""
    input_slice_path = os.path.join(DEBUG_TEMP_DIR, f"input_slice.png")
    output_slice_path = os.path.join(DEBUG_TEMP_DIR, f"seg_input_slice.png")
    output_path = os.path.join(DEBUG_TEMP_DIR)

    device = "cuda:0"
    medsam_model = sam_model_registry["vit_b"](checkpoint="work_dir/MedSAM/medsam_vit_b.pth")
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    image_embedding, box_1024, H, W = transform_image(input_array, medsam_model, bounding_box)
    segmented_slice = MedSAM_Inference.medsam_inference(medsam_model, image_embedding, box_1024, H, W)

    return np.array(segmented_slice), input_array


def transform_image(input_array, medsam_model, bounding_box=None):
    if len(input_array.shape) == 2:
        input_array = np.repeat(input_array[:, :, None], 3, axis=-1)

    H, W, _ = input_array.shape
    img_1024 = transform.resize(
        input_array, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)

    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to('cuda:0')
    )
    bounding_box = np.array(bounding_box).reshape(1, 4)
    box_1024 = bounding_box / np.array([W, H, W, H]) * 1024

    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)
    return image_embedding, box_1024, H, W


def process_volume(input_volume_path, output_volume_path, superpixel_volume_paths,
                   foreground_volume_path, perform_morphological_operations=True, ignore_background=True):
    print(f"\n[INFO] Processing volume: {input_volume_path}")
    img = nib.load(input_volume_path)
    volume_data = img.get_fdata()
    affine = img.affine
    header = img.header

    foreground_volume = nib.load(foreground_volume_path).get_fdata()

    # Load superpixel volumes
    superpixel_volumes = []
    for i, superpixel_volume_path in enumerate(superpixel_volume_paths):
        superpixel_img = nib.load(superpixel_volume_path)
        superpixel_volumes.append(superpixel_img.get_fdata())

    segmented_slices = []
    for i in range(volume_data.shape[2]):
        slice_2d = volume_data[..., i]
        foreground_slice = foreground_volume[..., i]
        print(f"\n[INFO] Processing slice {i + 1}/{volume_data.shape[2]}")

        segmented_slice = np.zeros_like(slice_2d)
        segmented_slice_temp = np.zeros_like(slice_2d)
        for j, superpixel_volume in enumerate(superpixel_volumes):
            superpixel_slice = superpixel_volume[..., i]
            superpixel_labels = np.unique(superpixel_slice)
            segmented_slice_temp = np.zeros_like(slice_2d)
            num_pixels = superpixel_slice.shape[0] * superpixel_slice.shape[1]

            print(f"[INFO] Processing superpixel level {j + 1} [{len(superpixel_labels) - 1} superpixels]")
            for label in superpixel_labels:
                label = int(label)
                if label == 0:
                    continue

                print(f".", end=" ")

                margin = 0
                bounding_box = np.argwhere(superpixel_slice == label)
                height, width = superpixel_slice.shape

                if len(bounding_box) == 0:
                    print(f"{label} no bounding box", end=" ")
                    continue

                x_min = max(0, bounding_box[:, 0].min() - margin)
                y_min = max(0, bounding_box[:, 1].min() - margin)
                x_max = min(width - 1, bounding_box[:, 0].max() + margin)
                y_max = min(height - 1, bounding_box[:, 1].max() + margin)

                bounding_box = [y_min, x_min, y_max - y_min, x_max - x_min]

                bounding_box_size = (x_max - x_min) * (y_max - y_min)
                if bounding_box_size > num_pixels / max(1, (4 * j)):
                    print(f"{label} too large ({bounding_box_size})", end=" ")
                    continue

                segmented_slice_label, transformed_image = segment_slice_with_sam(slice_2d, bounding_box=bounding_box)

                num_pixels_mask = np.count_nonzero(segmented_slice_label)
                # if num_pixels_mask < num_pixels / (40 * (j + 1)):
                if num_pixels_mask < 100:
                    print(f"{label} mask too small ({num_pixels_mask})", end=" ")
                    continue

                if perform_morphological_operations:
                    # circle kernel size 5x5
                    kernel = np.array([[0, 1, 0],
                                       [1, 1, 1],
                                       [0, 1, 0]], dtype=np.uint8)
                    segmented_slice_label = cv2.morphologyEx(segmented_slice_label.astype(np.uint8), cv2.MORPH_CLOSE,
                                                             kernel, iterations=5)

                    # only keep the largest connected component
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmented_slice_label,
                                                                                            connectivity=8)
                    max_label = 0
                    max_size = 0
                    for k in range(1, num_labels):
                        if stats[k, cv2.CC_STAT_AREA] > max_size:
                            max_label = k
                            max_size = stats[k, cv2.CC_STAT_AREA]
                    segmented_slice_label = np.where(labels == max_label, 1, 0)

                if ignore_background:
                    # if the mask fills mostly the background of the image, ignore it
                    # count nonzero pixels in the image where the mask is 1
                    num_zeros_under_mask = np.count_nonzero(
                        np.where((segmented_slice_label == 1) & (foreground_slice == 0), 1, 0))
                    if num_zeros_under_mask > 0.5 * num_pixels_mask:
                        print(f"{label} mask mostly background ({num_zeros_under_mask})", end=" ")
                        continue

                segmented_slice_label[segmented_slice_label == 1] = label
                segmented_slice_temp = np.where(segmented_slice_label != 0, segmented_slice_label, segmented_slice_temp)
                #
                # plt.figure()
                # plt.imshow(segmented_slice_temp)
                # plt.show()

            print(f"\n")

            max_value_segmented_slice = np.max(segmented_slice)
            segmented_slice_temp = np.where(segmented_slice_temp != 0, segmented_slice_temp + max_value_segmented_slice,
                                            segmented_slice_temp)
            #
            # plt.figure()
            # plt.imshow(slice_2d, cmap="gray")
            # plt.imshow(segmented_slice_temp, alpha=0.5)
            # plt.show()

        # get new labels so they go from 0 to n and dont skip any number
        unique_labels = np.unique(segmented_slice_temp)
        new_labels = np.arange(len(unique_labels))
        for old_label, new_label in zip(unique_labels, new_labels):
            segmented_slice_temp[segmented_slice_temp == old_label] = new_label
        segmented_slices.append(segmented_slice_temp)

    segmented_volume = np.stack(segmented_slices, axis=-1)
    print(f"[INFO] Segmented volume unique values: {np.unique(segmented_volume)}")

    segmented_img = nib.Nifti1Image(segmented_volume, affine, header=header)
    nib.save(segmented_img, output_volume_path)
    print(f"[INFO] Saved segmented volume to: {output_volume_path}")


if __name__ == "__main__":
    # dummy_volume_path = r"C:\Users\bonese\Documents\Courses\LRDL\LRDL-project\Self-supervised-Fewshot-Medical-Image-Segmentation\data\CHAOSCT\cropped\image_1-cropped.nii.gz"
    # dummy_superpixel_large_path = r"C:\Users\bonese\Documents\Courses\LRDL\LRDL-project\Self-supervised-Fewshot-Medical-Image-Segmentation\data\CHAOSCT\cropped\superpix-XLARGE_1-cropped.nii.gz"
    # dummy_superpixel_middle_path = r"C:\Users\bonese\Documents\Courses\LRDL\LRDL-project\Self-supervised-Fewshot-Medical-Image-Segmentation\data\CHAOSCT\cropped\superpix-MIDDLE_1-cropped.nii.gz"
    # dummy_superpixel_small_path = r"C:\Users\bonese\Documents\Courses\LRDL\LRDL-project\Self-supervised-Fewshot-Medical-Image-Segmentation\data\CHAOSCT\cropped\superpix-SMALL_1-cropped.nii.gz"
    # dummy_superpixel_paths = [dummy_superpixel_large_path, dummy_superpixel_middle_path, dummy_superpixel_small_path]
    # dummy_output_path = r"C:\Users\bonese\Documents\Courses\LRDL\LRDL-project\Self-supervised-Fewshot-Medical-Image-Segmentation\data\CHAOSCT\cropped\segmented_1-cropped.nii.gz"
    # process_volume(dummy_volume_path, dummy_output_path, dummy_superpixel_paths)

    inputs = r"C:\Users\bonese\Documents\Courses\LRDL\LRDL-project\Self-supervised-Fewshot-Medical-Image-Segmentation\data\CHAOSMR\chaos_MR_T2_normalized"

    for input in os.listdir(inputs):
        if input.startswith("image"):
            input_volume_path = os.path.join(inputs, input)
            num = input.split("_")[-1].split(".")[0]
            superpixel_large = os.path.join(inputs, f"superpix-XLARGE_{num}.nii.gz")
            superpixel_middle = os.path.join(inputs, f"superpix-MIDDLE_{num}.nii.gz")
            superpixel_small = os.path.join(inputs, f"superpix-SMALL_{num}.nii.gz")
            # superpixel_volumes = [superpixel_large, superpixel_middle, superpixel_small]
            superpixel_volumes = [superpixel_middle]
            foreground_volume = os.path.join(inputs, f"fgmask_{num}.nii.gz")
            output_volume_path = os.path.join(inputs, f"segmented_{num}.nii.gz")
            process_volume(input_volume_path, output_volume_path, superpixel_volumes, foreground_volume)
