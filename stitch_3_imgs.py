import cv2
import numpy as np
import sys

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
    
    # Calculate mean brightness
    mean_tgt = np.mean(hsv_tgt[:,:,2])
    mean_ref = np.mean(hsv_ref[:,:,2])
    
    # Avoid division by zero
    if mean_tgt == 0: return target_img
    
    # Calculate gain and clip it to avoid extreme changes
    gain = mean_ref / mean_tgt
    gain = np.clip(gain, 0.5, 2.0)
    
    # Apply gain to V channel
    v_channel = hsv_tgt[:,:,2].astype(np.float32) * gain
    hsv_tgt[:,:,2] = np.clip(v_channel, 0, 255).astype(np.uint8)
    
    print(f"  > Exposure fixed: gain {gain:.2f}")
    return cv2.cvtColor(hsv_tgt, cv2.COLOR_HSV2BGR)

def resize_img(img, width=800):
    """Resize image to a manageable size to save memory."""
    h, w = img.shape[:2]
    scale = width / w
    return cv2.resize(img, (int(w*scale), int(h*scale)))

# ==========================================
#  Core Logic: Feature Matching & Homography
# ==========================================

def get_homography(img1, img2, sift):
    """
    Detect SIFT features, match them using KNN, 
    and compute the Homography matrix using RANSAC.
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
        return H
    else:
        print("Warning: Not enough matches found!")
        return None

# ==========================================
#  Advanced Blending: Distance Transform
# ==========================================

def create_weight_map(img):
    """
    Create a weight map for blending.
    Pixels closer to the center get higher weights (1.0),
    pixels near the black edge get lower weights (-> 0).
    """
    # Generate binary mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Calculate distance to the nearest zero pixel
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # Normalize to 0-1 range
    cv2.normalize(dist_map, dist_map, 0, 1.0, cv2.NORM_MINMAX)
    
    return dist_map

# ==========================================
#  Main Pipeline
# ==========================================

def stitch_images(left_img, mid_img, right_img):
    print("=== Start Stitching ===")
    
    # --- Step 0: Pre-processing ---
    print("[1/6] Compensating exposure...")
    left_img = compensate_exposure(left_img, mid_img)
    right_img = compensate_exposure(right_img, mid_img)

    # --- Step 1: Feature Detection ---
    print("[2/6] Initializing SIFT...")
    sift = cv2.SIFT_create()
    
    print("[3/6] Computing Homography...")
    H_left = get_homography(left_img, mid_img, sift)
    H_right = get_homography(right_img, mid_img, sift)
    
    if H_left is None or H_right is None:
        print("Error: Stitching failed due to bad matching.")
        return None

    # --- Step 2: Calculate Canvas Size ---
    # We need to find the size of the final panorama to avoid cropping
    h, w, _ = mid_img.shape
    h_l, w_l, _ = left_img.shape
    h_r, w_r, _ = right_img.shape
    
    # Transform corners of left and right images
    pts_left = np.float32([[0,0], [0,h_l], [w_l,h_l], [w_l,0]]).reshape(-1,1,2)
    dst_left = cv2.perspectiveTransform(pts_left, H_left)
    
    pts_right = np.float32([[0,0], [0,h_r], [w_r,h_r], [w_r,0]]).reshape(-1,1,2)
    dst_right = cv2.perspectiveTransform(pts_right, H_right)
    
    pts_mid = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)

    # Get the bounding box
    all_pts = np.concatenate((dst_left, pts_mid, dst_right), axis=0)
    [x_min, y_min] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_pts.max(axis=0).ravel() + 0.5)
    
    # Translation matrix to shift images to positive coordinates
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], 
                              [0, 1, translation_dist[1]], 
                              [0, 0, 1]], dtype=np.float32)
    output_shape = (x_max - x_min, y_max - y_min)

    # --- Step 3: Warping ---
    print("[4/6] Warping images...")
    warped_left = cv2.warpPerspective(left_img, H_translation.dot(H_left), output_shape)
    warped_right = cv2.warpPerspective(right_img, H_translation.dot(H_right), output_shape)
    warped_mid = cv2.warpPerspective(mid_img, H_translation, output_shape)

    # --- Step 4: Blending (Weighted) ---
    print("[5/6] Blending images (Distance Weighted)...")
    
    # Compute weight maps for each warped image
    w_left = create_weight_map(warped_left)
    w_mid = create_weight_map(warped_mid)
    w_right = create_weight_map(warped_right)
    
    # Sum of weights
    w_sum = w_left + w_mid + w_right
    w_sum[w_sum == 0] = 0.00001  # Avoid division by zero
    
    # Merge weights to 3 channels
    w_left_3ch = cv2.merge([w_left, w_left, w_left])
    w_mid_3ch = cv2.merge([w_mid, w_mid, w_mid])
    w_right_3ch = cv2.merge([w_right, w_right, w_right])
    w_sum_3ch = cv2.merge([w_sum, w_sum, w_sum])
    
    # Calculate final blended image (Float32 for precision)
    blended_float = (warped_left.astype(np.float32) * w_left_3ch +
                     warped_mid.astype(np.float32) * w_mid_3ch +
                     warped_right.astype(np.float32) * w_right_3ch) / w_sum_3ch
                     
    final_result = np.clip(blended_float, 0, 255).astype(np.uint8)

    # --- Step 5: Post-processing (Crop black edges) ---
    print("[6/6] Auto-cropping...")
    gray = cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Add a small padding crop to remove edge artifacts
        padding = 2
        final_result = final_result[y+padding:y+h-padding, x+padding:x+w-padding]

    return final_result

if __name__ == '__main__':
    # Load images
    l = cv2.imread('images/left.jpg')
    m = cv2.imread('images/mid.jpg')
    r = cv2.imread('images/right.jpg')
    
    if l is None or m is None or r is None:
        print("Error: Images not found. Check 'images' folder.")
    else:
        # Resize for speed
        l = resize_img(l)
        m = resize_img(m)
        r = resize_img(r)
        
        # Run stitching
        result = stitch_images(l, m, r)
        
        if result is not None:
            cv2.imshow('Result', result)
            cv2.imwrite('result.jpg', result)
            print("\n✅ Success! Saved as 'result.jpg'.")
            print("Press any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()