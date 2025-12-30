import cv2
import numpy as np
import sys
import os

# ==========================================
#  Helper Functions & Pre-processing
# ==========================================

def compensate_exposure(target_img, ref_img):
    hsv_tgt = cv2.cvtColor(target_img, cv2.COLOR_BGR2HSV)
    hsv_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
    mean_tgt = np.mean(hsv_tgt[:,:,2])
    mean_ref = np.mean(hsv_ref[:,:,2])
    if mean_tgt == 0: return target_img
    gain = mean_ref / mean_tgt
    gain = np.clip(gain, 0.5, 2.0)
    v_channel = hsv_tgt[:,:,2].astype(np.float32) * gain
    hsv_tgt[:,:,2] = np.clip(v_channel, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv_tgt, cv2.COLOR_HSV2BGR)

def resize_img(img, width=800):
    h, w = img.shape[:2]
    scale = width / w
    return cv2.resize(img, (int(w*scale), int(h*scale)))

def get_crop_coords(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None: return 0, img.shape[0], 0, img.shape[1]
    x, y, w, h = cv2.boundingRect(coords)
    img_h, img_w = img.shape[:2]
    threshold_percent = 0.8  
    top, bottom, left, right = y, y+h, x, x+w
    for r in range(y, y+h):
        if np.count_nonzero(thresh[r, x:x+w]) / w > threshold_percent:
            top = r
            break
    for r in range(y+h-1, y-1, -1):
        if np.count_nonzero(thresh[r, x:x+w]) / w > threshold_percent:
            bottom = r
            break
    for c in range(x, x+w):
        if np.count_nonzero(thresh[top:bottom, c]) / (bottom-top) > threshold_percent:
            left = c
            break
    for c in range(x+w-1, x-1, -1):
        if np.count_nonzero(thresh[top:bottom, c]) / (bottom-top) > threshold_percent:
            right = c
            break
    area_bbox = w * h
    crop_w = right - left
    crop_h = bottom - top
    area_crop = crop_w * crop_h
    if area_crop < 0.6 * area_bbox:
        padding_ratio = 0.02
        safe_top = y + int(h * padding_ratio)
        safe_bottom = y + h - int(h * padding_ratio)
        safe_left = x + int(w * padding_ratio)
        safe_right = x + w - int(w * padding_ratio)
        return safe_top, safe_bottom, safe_left, safe_right
    else:
        return top, bottom, left, right

# ==========================================
#  Core Logic: Feature Matching & Visualization
# ==========================================

def get_homography(img1, img2, sift, name="Pair", vis_save_path=None):
    """
    Computes Homography and optionally saves a visualization of matches.
    """
    # 1. Detect features
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # 2. Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # 3. Filter matches (Lowe's ratio test)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            
    # 4. Compute Homography if enough matches are found
    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # --- Stats Printing ---
        num_matches = len(good)
        num_inliers = int(mask.sum()) if mask is not None else 0
        inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
        print(f"    📊 [Stats {name}] Matches: {num_matches} | Inliers: {num_inliers} | Ratio: {inlier_ratio:.2%}")
        
        # --- Visualization: Draw Inliers ---
        if vis_save_path is not None:
            matchesMask = mask.ravel().tolist()
            draw_params = dict(matchColor=(0, 255, 0), # Green color for inliers
                               singlePointColor=None,
                               matchesMask=matchesMask, # Only draw inliers
                               flags=2) 
            img_vis = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
            cv2.imwrite(vis_save_path, img_vis)
        
        return H
    else:
        print(f"Warning: Not enough matches found for {name}!")
        return None

def create_weight_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    cv2.normalize(dist_map, dist_map, 0, 1.0, cv2.NORM_MINMAX)
    return dist_map

# ==========================================
#  Stitching Pipeline
# ==========================================

def stitch_images(left_img, mid_img, right_img, set_name, output_dir, match_dir):
    print(f"[{set_name}] Processing...")
    
    # 1. Pre-processing
    left_img = compensate_exposure(left_img, mid_img)
    right_img = compensate_exposure(right_img, mid_img)

    # 2. Matching
    sift = cv2.SIFT_create()
    
    
    vis_path_L = os.path.join(match_dir, f'vis_match_{set_name}_left_mid.jpg')
    vis_path_R = os.path.join(match_dir, f'vis_match_{set_name}_right_mid.jpg')
    
    # Pass paths to get_homography
    H_left = get_homography(left_img, mid_img, sift, name="Left->Mid", vis_save_path=vis_path_L)
    H_right = get_homography(right_img, mid_img, sift, name="Right->Mid", vis_save_path=vis_path_R)
    
    if H_left is None or H_right is None: return None

    # 3. Canvas
    h, w, _ = mid_img.shape
    h_l, w_l, _ = left_img.shape
    h_r, w_r, _ = right_img.shape
    pts_left = np.float32([[0,0], [0,h_l], [w_l,h_l], [w_l,0]]).reshape(-1,1,2)
    dst_left = cv2.perspectiveTransform(pts_left, H_left)
    pts_right = np.float32([[0,0], [0,h_r], [w_r,h_r], [w_r,0]]).reshape(-1,1,2)
    dst_right = cv2.perspectiveTransform(pts_right, H_right)
    pts_mid = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)

    all_pts = np.concatenate((dst_left, pts_mid, dst_right), axis=0)
    [x_min, y_min] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_pts.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], 
                              [0, 1, translation_dist[1]], 
                              [0, 0, 1]], dtype=np.float32)
    output_shape = (x_max - x_min, y_max - y_min)

    # 4. Warping
    warped_left = cv2.warpPerspective(left_img, H_translation.dot(H_left), output_shape)
    warped_right = cv2.warpPerspective(right_img, H_translation.dot(H_right), output_shape)
    warped_mid = cv2.warpPerspective(mid_img, H_translation, output_shape)

    # 5. Blending
    w_left = create_weight_map(warped_left)
    w_mid = create_weight_map(warped_mid)
    w_right = create_weight_map(warped_right)
    w_sum = w_left + w_mid + w_right
    w_sum[w_sum == 0] = 0.00001
    w_left_3ch = cv2.merge([w_left, w_left, w_left])
    w_mid_3ch = cv2.merge([w_mid, w_mid, w_mid])
    w_right_3ch = cv2.merge([w_right, w_right, w_right])
    w_sum_3ch = cv2.merge([w_sum, w_sum, w_sum])
    blended_float = (warped_left.astype(np.float32) * w_left_3ch +
                     warped_mid.astype(np.float32) * w_mid_3ch +
                     warped_right.astype(np.float32) * w_right_3ch) / w_sum_3ch
    final_result = np.clip(blended_float, 0, 255).astype(np.uint8)

    # Save Step 1
    cv2.imwrite(os.path.join(output_dir, f'result_{set_name}_step1_raw.jpg'), final_result)

    # 6. Auto-cropping
    top, bottom, left, right = get_crop_coords(final_result)
    
    # Save Step 2 (Red Box)
    vis_img = final_result.copy()
    cv2.rectangle(vis_img, (left, top), (right, bottom), (0, 0, 255), 5)
    cv2.imwrite(os.path.join(output_dir, f'result_{set_name}_step2_bbox.jpg'), vis_img)

    # Save Step 3 (Final)
    final_crop = final_result[top:bottom, left:right]
    cv2.imwrite(os.path.join(output_dir, f'result_{set_name}_step3_final.jpg'), final_crop)
    
    print(f"✅ [{set_name}] All steps saved.")

# ==========================================
#  Batch Execution
# ==========================================

if __name__ == '__main__':
    input_root = 'images'
    output_dir = 'result_images'
    match_dir = 'match_images'  
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(match_dir): os.makedirs(match_dir) 

    subfolders = [f.path for f in os.scandir(input_root) if f.is_dir()]
    subfolders.sort()
    print(f"Found {len(subfolders)} sets.\n")

    for folder in subfolders:
        set_name = os.path.basename(folder)
        path_l = os.path.join(folder, 'left.jpg')
        path_m = os.path.join(folder, 'mid.jpg')
        path_r = os.path.join(folder, 'right.jpg')
        
        if not (os.path.exists(path_l) and os.path.exists(path_m) and os.path.exists(path_r)):
            continue
            
        try:
            l = resize_img(cv2.imread(path_l))
            m = resize_img(cv2.imread(path_m))
            r = resize_img(cv2.imread(path_r))
           
            stitch_images(l, m, r, set_name, output_dir, match_dir)
        except Exception as e:
            print(f"❌ [{set_name}] Error: {str(e)}\n")

    print("\n=== Processing Complete ===")