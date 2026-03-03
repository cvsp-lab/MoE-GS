import os
import numpy as np
import cv2
import argparse

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 255.0
    psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr_value

def apply_mask_and_average_frames(render_dir, mask_dir):

    # render_dir: .../renders
    # mask_dir : .../mask/cam00
    render_files = os.listdir(render_dir) 

    masked_renderes = []
    for _, filename in enumerate(sorted(render_files)):
        if filename.endswith('.png') or filename.endswith('.jpg'):  # assuming frames are in PNG or JPG format
            rendered_path = os.path.join(render_dir, filename)
            renderer = cv2.imread(rendered_path)
            
            # 독립 마스크
            # mask_path = os.path.join(mask_dir, str(idx+start_iteration).zfill(4) + '.png')
            # mask_path = os.path.join(mask_dir, 'common_mask_threshold_1.00.png')
            mask_path = os.path.join(mask_dir, 'common_mask.png')


            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
            resized_mask = cv2.resize(mask, (renderer.shape[1], renderer.shape[0]), interpolation=cv2.INTER_NEAREST)

            masked_renderer = renderer * resized_mask[:, :, np.newaxis]  # Apply mask across all color channels
            cv2.imwrite('masked_renderer.png', masked_renderer)

            masked_renderes.append(masked_renderer)

    average_renderer = np.mean(masked_renderes, axis=0).astype(np.uint8)

    return average_renderer, masked_renderes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Dataset path")
    # parser.add_argument("--output_dir", type=str, help="...test/ours_14000")
    parser.add_argument("--output_dir", type=str, help="...test/ours_50000")
    # Common mask 사용
    # parser.add_argument("--start_iteration", type=int, help=" 50-99 frame")

    args = parser.parse_args()
    
    data_dir = args.dataset_path
    mask_dir = os.path.join(data_dir,'mask', 'cam00')
    # mask_dir = "/home/cvsp/ETRI_2024/mask/common_mask2_cook_spanish"

    render_dir = os.path.join(args.output_dir, 'renders')
    # gt_dir = os.path.join(args.output_dir, 'gt')

    average_renderer, masked_renderes = apply_mask_and_average_frames(render_dir, mask_dir)

    cv2.imwrite('average_renderer.png', average_renderer)

    psnr_scores = []
    for idx, masked_renderer in enumerate(masked_renderes):
        psnr_value = psnr(masked_renderer, average_renderer)
        psnr_scores.append(psnr_value)

    average_psnr = np.mean(psnr_scores)
    print(f"Average PSNR: {average_psnr:.2f} dB")
