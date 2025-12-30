import cv2
import numpy as np
import sys
import os  

# ==========================================
#  Helper Functions & Pre-processing
# ==========================================

def compensate_exposure(target_img, ref_img):
    """
    Simple exposure compensation.
    Adjust the brightness (V channel in HSV) of the target image 
    to match the reference image. Helps with the "seam" problem.
    """
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
    """Resize image to a manageable size."""
    h, w = img.shape[:2]
    scale = width / w
    return cv2.resize(img, (int(w*scale), int(h*scale)))

# ==========================================
#  Core Logic: Feature Matching & Homography
# ==========================================

def get_homography(img1, img2, sift):
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            
    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    else:
        return None

# ==========================================
#  Advanced Blending: Distance Transform
# ==========================================

def create_weight_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    cv2.normalize(dist_map, dist_map, 0, 1.0, cv2.NORM_MINMAX)
    return dist_map

# ==========================================
#  Main Pipeline (Single Set)
# ==========================================

def stitch_images(left_img, mid_img, right_img, set_name="Unknown"):
    print(f"[{set_name}] 1. Pre-processing...")
    left_img = compensate_exposure(left_img, mid_img)
    right_img = compensate_exposure(right_img, mid_img)

    print(f"[{set_name}] 2. Matching Features...")
    sift = cv2.SIFT_create()
    H_left = get_homography(left_img, mid_img, sift)
    H_right = get_homography(right_img, mid_img, sift)
    
    if H_left is None or H_right is None:
        print(f"[{set_name}] Error: Not enough matches.")
        return None

    # Calculate Canvas
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

    print(f"[{set_name}] 3. Warping...")
    warped_left = cv2.warpPerspective(left_img, H_translation.dot(H_left), output_shape)
    warped_right = cv2.warpPerspective(right_img, H_translation.dot(H_right), output_shape)
    warped_mid = cv2.warpPerspective(mid_img, H_translation, output_shape)

    print(f"[{set_name}] 4. Blending...")
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

    # Auto-cropping
    print(f"[{set_name}] 5. Cropping...")
    gray = cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((21, 21), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        pad_x, pad_y = 20, 85
        x2, y2 = max(x + pad_x, 0), max(y + pad_y, 0)
        w2, h2 = max(w - 2 * pad_x, 1), max(h - 2 * pad_y, 1)
        final_result = final_result[y2:y2+h2, x2:x2+w2]

    return final_result

# ==========================================
#  Batch Processing Main Logic
# ==========================================

if __name__ == '__main__':
    
    input_root = 'images'
    output_dir = 'result_images'
    
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")


    subfolders = [f.path for f in os.scandir(input_root) if f.is_dir()]
    
    if not subfolders:
        print(f"Warning: No folders found inside '{input_root}'. Check your structure!")
    
    print(f"Found {len(subfolders)} sets of images. Starting batch processing...\n")

   
    for folder in subfolders:
        set_name = os.path.basename(folder) 
        
       
        path_l = os.path.join(folder, 'left.jpg')
        path_m = os.path.join(folder, 'mid.jpg')
        path_r = os.path.join(folder, 'right.jpg')
        
        
        if not (os.path.exists(path_l) and os.path.exists(path_m) and os.path.exists(path_r)):
            print(f"❌ Skipping [{set_name}]: Missing images (needs left.jpg, mid.jpg, right.jpg)")
            continue
            
        
        l = resize_img(cv2.imread(path_l))
        m = resize_img(cv2.imread(path_m))
        r = resize_img(cv2.imread(path_r))
        
        
        try:
            result = stitch_images(l, m, r, set_name)
            
            if result is not None:
                
                save_path = os.path.join(output_dir, f'result_{set_name}.jpg')
                cv2.imwrite(save_path, result)
                print(f"✅ [{set_name}] Success! Saved to: {save_path}\n")
            else:
                print(f"❌ [{set_name}] Failed: Algorithm returned None.\n")
                
        except Exception as e:
            print(f"❌ [{set_name}] Crashed: {str(e)}\n")

    print("=== Batch Processing Complete ===")