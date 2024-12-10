import os
import cv2
import numpy as np

dataset_path = 'dataset'
good = os.path.join(dataset_path, 'good')
bad = os.path.join(dataset_path, 'bad')
masks = os.path.join(dataset_path, 'masks')

bad_images = os.listdir(bad)
good_images = os.listdir(good)
masks_images = os.listdir(masks)

good_count = len(good_images)
bad_count = len(bad_images)
remaining_images = good_count - bad_count

print(f"Good images: {good_count}, Bad images: {bad_count}")
print(f"Remaining images to generate: {remaining_images}")

for i in range(remaining_images):
    bad_index = i % bad_count
    mask_index = bad_index  

    bad_images_path = os.path.join(bad, bad_images[bad_index])
    masks_images_path = os.path.join(masks, masks_images[mask_index])
    good_images_path = os.path.join(good, good_images[i % good_count])  

    g_img = cv2.imread(good_images_path)
    b_img = cv2.imread(bad_images_path)
    m_img = cv2.imread(masks_images_path)

    g_img = cv2.resize(g_img, (b_img.shape[1], b_img.shape[0]))

    alpha = 0.5  
    blended_img = cv2.addWeighted(g_img, alpha, b_img, 1 - alpha, 0)

    blended_filename = f"blended_{i + bad_count}.png"
    blended_path = os.path.join(bad, blended_filename)
    mask_path = os.path.join(masks, blended_filename)  

    cv2.imwrite(blended_path, blended_img)
    cv2.imwrite(mask_path, m_img)

    print(f"Generated: {blended_path} with mask: {mask_path}")

print("Image blending complete. Bad image count now matches good image count.")