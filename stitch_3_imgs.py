import cv2
import numpy as np
import sys

# ==========================================
#  [Module 1] 光度处理 & 辅助函数
# ==========================================

def compensate_exposure(target_img, ref_img):
    """[关键改进] 保持不变：曝光补偿"""
    hsv_tgt = cv2.cvtColor(target_img, cv2.COLOR_BGR2HSV)
    hsv_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
    mean_tgt = np.mean(hsv_tgt[:,:,2])
    mean_ref = np.mean(hsv_ref[:,:,2])
    if mean_tgt == 0: return target_img
    gain = mean_ref / mean_tgt
    gain = np.clip(gain, 0.5, 2.0)
    v_channel = hsv_tgt[:,:,2].astype(np.float32) * gain
    hsv_tgt[:,:,2] = np.clip(v_channel, 0, 255).astype(np.uint8)
    print(f"  > 曝光补偿: 亮度增益 {gain:.2f}")
    return cv2.cvtColor(hsv_tgt, cv2.COLOR_HSV2BGR)

def resize_img(img, width=800):
    """辅助函数：统一缩放"""
    h, w = img.shape[:2]
    scale = width / w
    return cv2.resize(img, (int(w*scale), int(h*scale)))

# ==========================================
#  [Module 2] 几何变换核心
# ==========================================

def get_homography(img1, img2, sift):
    """计算单应性矩阵 H"""
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
#  [Module 3] 新版融合核心 (Weighted Blending)
# ==========================================

def create_weight_map(img):
    """
    [核心改进] 创建权重图
    计算每个有效像素到最近黑色边缘的距离。距离越远，权重越大。
    这样重叠区域中心部分的像素会起主导作用，边缘部分则逐渐淡出。
    """
    # 1. 生成二值 Mask (有像素为255，黑色背景为0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # 2. 计算距离变换 (Distance Transform)
    # dist_map 的值是像素到最近 0 值像素的欧氏距离
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # 3. 归一化到 0~1 之间
    cv2.normalize(dist_map, dist_map, 0, 1.0, cv2.NORM_MINMAX)
    
    return dist_map

def stitch_3_images_v3(left_img, mid_img, right_img):
    print("=== 开始终极版拼接 (v3) ===")
    
    # --- Step 0: 预处理 - 曝光补偿 ---
    print("[1/6] 曝光补偿...")
    left_img = compensate_exposure(left_img, mid_img)
    right_img = compensate_exposure(right_img, mid_img)

    print("[2/6] 初始化 SIFT...")
    sift = cv2.SIFT_create()
    
    # --- Step 1: 计算变换矩阵 ---
    print("[3/6] 计算单应性矩阵...")
    H_left = get_homography(left_img, mid_img, sift)
    H_right = get_homography(right_img, mid_img, sift)
    if H_left is None or H_right is None: return None

    # --- Step 2: 计算画布大小 ---
    h, w, _ = mid_img.shape
    h_l, w_l, _ = left_img.shape
    pts_left = np.float32([[0,0], [0,h_l], [w_l,h_l], [w_l,0]]).reshape(-1,1,2)
    dst_left = cv2.perspectiveTransform(pts_left, H_left)
    h_r, w_r, _ = right_img.shape
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

    # --- Step 3: 图像变形 (Warping) ---
    print("[4/6] 图像变形...")
    warped_left = cv2.warpPerspective(left_img, H_translation.dot(H_left), output_shape)
    warped_right = cv2.warpPerspective(right_img, H_translation.dot(H_right), output_shape)
    warped_mid = cv2.warpPerspective(mid_img, H_translation, output_shape)

    # --- Step 4: [核心改进] 并行加权融合 ---
    print("[5/6] 执行并行加权融合 (Weighted Blending)...")
    
    # 1. 计算三张图各自的权重图
    w_left = create_weight_map(warped_left)
    w_mid = create_weight_map(warped_mid)
    w_right = create_weight_map(warped_right)
    
    # 2. 计算总权重图 (用于归一化)
    w_sum = w_left + w_mid + w_right
    # 避免除以0，把0替换成一个很小的数
    w_sum[w_sum == 0] = 0.00001
    
    # 3. 将权重图扩展为 3 通道以便与图像相乘
    w_left_3ch = cv2.merge([w_left, w_left, w_left])
    w_mid_3ch = cv2.merge([w_mid, w_mid, w_mid])
    w_right_3ch = cv2.merge([w_right, w_right, w_right])
    w_sum_3ch = cv2.merge([w_sum, w_sum, w_sum])
    
    # 4. 加权平均计算 (核心公式)
    # 使用 float32 进行高精度计算
    blended_float = (warped_left.astype(np.float32) * w_left_3ch +
                     warped_mid.astype(np.float32) * w_mid_3ch +
                     warped_right.astype(np.float32) * w_right_3ch) / w_sum_3ch
                     
    # 5. 转回 uint8
    final_result = np.clip(blended_float, 0, 255).astype(np.uint8)

    # --- Step 5: 裁剪黑边 ---
    print("[6/6] 最终裁剪...")
    gray = cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY)
    # 使用一个稍微高一点的阈值来去除可能的黑色噪点
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # 稍微往里收缩一点，确保切掉边缘的瑕疵
        padding = 2
        final_result = final_result[y+padding:y+h-padding, x+padding:x+w-padding]

    return final_result

if __name__ == '__main__':
    l = cv2.imread('images/left.jpg')
    m = cv2.imread('images/mid.jpg')
    r = cv2.imread('images/right.jpg')
    
    if l is None or m is None or r is None:
        print("错误：请检查 images 文件夹")
    else:
        l = resize_img(l)
        m = resize_img(m)
        r = resize_img(r)
        
        # 运行终极版拼接
        result = stitch_3_images_v3(l, m, r)
        
        if result is not None:
            cv2.imshow('Final V3 Panorama', result)
            cv2.imwrite('result_v3_perfect.jpg', result)
            print("\n✅ 拼接成功！结果已保存为 result_v3_perfect.jpg")
            print("这个版本应该彻底消除了突兀的拼接痕迹。")
            cv2.waitKey(0)
            cv2.destroyAllWindows()