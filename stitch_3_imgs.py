import cv2
import numpy as np
import sys

# ==========================================
#  👇 Part 1: 拉普拉斯金字塔融合核心工具包
# ==========================================

def gaussian_pyramid(img, levels):
    """构建高斯金字塔"""
    pyr = [img]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        pyr.append(img)
    return pyr

def laplacian_pyramid(gauss_pyr):
    """构建拉普拉斯金字塔"""
    pyr = []
    for i in range(len(gauss_pyr) - 1):
        img_expanded = cv2.pyrUp(gauss_pyr[i+1])
        h, w = gauss_pyr[i].shape[:2]
        img_expanded = cv2.resize(img_expanded, (w, h))
        
        # 拉普拉斯层 = 当前层 - 模糊后的上一层
        lap = cv2.subtract(gauss_pyr[i], img_expanded)
        pyr.append(lap)
    pyr.append(gauss_pyr[-1])
    return pyr

def reconstruct(lap_pyr):
    """图像重建"""
    img = lap_pyr[-1]
    for i in range(len(lap_pyr) - 2, -1, -1):
        img_expanded = cv2.pyrUp(img)
        h, w = lap_pyr[i].shape[:2]
        img_expanded = cv2.resize(img_expanded, (w, h))
        img = cv2.add(lap_pyr[i], img_expanded)
    return img

def laplacian_blend(img1, img2, mask, levels=4):
    """
    金字塔融合主函数 (修正版：强制使用 float64 避免细节丢失)
    """
    # ⚠️ 关键修改：在建立金字塔前，强制转为 float64
    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)
    mask_f = mask.astype(np.float64)

    gauss_pyr_img1 = gaussian_pyramid(img1_f, levels)
    gauss_pyr_img2 = gaussian_pyramid(img2_f, levels)
    gauss_pyr_mask = gaussian_pyramid(mask_f, levels)

    lap_pyr_img1 = laplacian_pyramid(gauss_pyr_img1)
    lap_pyr_img2 = laplacian_pyramid(gauss_pyr_img2)

    blend_pyr = []
    for l1, l2, m in zip(lap_pyr_img1, lap_pyr_img2, gauss_pyr_mask):
        if len(m.shape) == 2:
            m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        
        # 核心加权计算
        ls = l1 * m + l2 * (1.0 - m)
        blend_pyr.append(ls)

    # 重建后结果仍为 float64
    return reconstruct(blend_pyr)

# ==========================================
#  👇 Part 2: 基础几何辅助函数
# ==========================================

def resize_img(img, width=800):
    """调整图片大小，防止内存溢出"""
    h, w = img.shape[:2]
    scale = width / w
    return cv2.resize(img, (int(w*scale), int(h*scale)))

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
#  👇 Part 3: 主拼接逻辑 (几何+融合)
# ==========================================

def stitch_3_images_advanced(left_img, mid_img, right_img):
    print("[1/4] 初始化 SIFT...")
    sift = cv2.SIFT_create()
    
    # --- Step 1: 计算变换矩阵 ---
    print("[2/4] 计算单应性矩阵 (Homography)...")
    H_left = get_homography(left_img, mid_img, sift)
    H_right = get_homography(right_img, mid_img, sift)
    
    if H_left is None or H_right is None:
        print("错误：特征匹配点不足，无法拼接！")
        return None

    # --- Step 2: 计算画布大小 ---
    h, w, _ = mid_img.shape
    
    # 左图变换后的角点
    h_l, w_l, _ = left_img.shape
    pts_left = np.float32([[0,0], [0,h_l], [w_l,h_l], [w_l,0]]).reshape(-1,1,2)
    dst_left = cv2.perspectiveTransform(pts_left, H_left)

    # 右图变换后的角点
    h_r, w_r, _ = right_img.shape
    pts_right = np.float32([[0,0], [0,h_r], [w_r,h_r], [w_r,0]]).reshape(-1,1,2)
    dst_right = cv2.perspectiveTransform(pts_right, H_right)

    # 中图角点
    pts_mid = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)

    # 计算最大画布范围
    all_pts = np.concatenate((dst_left, pts_mid, dst_right), axis=0)
    [x_min, y_min] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_pts.max(axis=0).ravel() + 0.5)
    
    # 平移矩阵
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], 
                              [0, 1, translation_dist[1]], 
                              [0, 0, 1]], dtype=np.float32)

    # --- Step 3: 图像变形 (Warping) ---
    print("[3/4] 图像变形与对齐...")
    output_shape = (x_max - x_min, y_max - y_min)
    
    warped_left = cv2.warpPerspective(left_img, H_translation.dot(H_left), output_shape)
    warped_right = cv2.warpPerspective(right_img, H_translation.dot(H_right), output_shape)
    warped_mid = cv2.warpPerspective(mid_img, H_translation, output_shape)

    # --- Step 4: 进阶融合 (Pyramid Blending) ---
    print("[4/4] 正在进行金字塔融合 (消除接缝)...")

    # === A. 融合 Left 和 Mid ===
    mask_left_binary = cv2.cvtColor(warped_left, cv2.COLOR_BGR2GRAY)
    _, mask_left_binary = cv2.threshold(mask_left_binary, 1, 255, cv2.THRESH_BINARY)
    mask_left_float = mask_left_binary.astype(np.float32) / 255.0
    
    # 模糊掩膜
    mask_left_blurred = cv2.GaussianBlur(mask_left_float, (201, 201), 0)
    mask_3ch = cv2.merge([mask_left_blurred, mask_left_blurred, mask_left_blurred])
    
    # 调用融合 (结果是 float64)
    blend_LM = laplacian_blend(warped_left, warped_mid, mask_3ch, levels=6)
    
    # ⚠️ 关键步骤：中间结果转回 uint8 以便下一步处理，或者保持 float 继续做
    # 这里为了简单，我们先转回 uint8 方便生成 mask
    blend_LM_uint8 = np.clip(blend_LM, 0, 255).astype(np.uint8)

    # === B. 融合 (Left+Mid) 和 Right ===
    mask_right_binary = cv2.cvtColor(warped_right, cv2.COLOR_BGR2GRAY)
    _, mask_right_binary = cv2.threshold(mask_right_binary, 1, 255, cv2.THRESH_BINARY)
    mask_right_inv = cv2.bitwise_not(mask_right_binary)
    mask_right_float = mask_right_inv.astype(np.float32) / 255.0
    
    # 模糊
    mask_right_blurred = cv2.GaussianBlur(mask_right_float, (201, 201), 0)
    mask_right_3ch = cv2.merge([mask_right_blurred, mask_right_blurred, mask_right_blurred])
    
    # 融合 (注意这里输入 blend_LM_uint8)
    final_result_float = laplacian_blend(blend_LM_uint8, warped_right, mask_right_3ch, levels=4)
    
    # ⚠️ 最终输出必须转回 0-255 的整数
    final_result = np.clip(final_result_float, 0, 255).astype(np.uint8)

    # --- Step 5: 自动裁剪黑边 ---
    gray = cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        final_result = final_result[y:y+h, x:x+w]

    return final_result

# ==========================================
#  👇 程序入口
# ==========================================
if __name__ == '__main__':
    # 读取图片
    l = cv2.imread('images/left.jpg')
    m = cv2.imread('images/mid.jpg')
    r = cv2.imread('images/right.jpg')
    
    if l is None or m is None or r is None:
        print("错误：找不到图片，请确保 images 文件夹下有 left.jpg, mid.jpg, right.jpg")
    else:
        # 统一调整大小
        l = resize_img(l)
        m = resize_img(m)
        r = resize_img(r)
        
        # 运行
        result = stitch_3_images_advanced(l, m, r)
        
        if result is not None:
            cv2.imshow('Final Panorama', result)
            cv2.imwrite('result_advanced.jpg', result)
            print("✅ 拼接成功！结果已保存为 result_advanced.jpg")
            print("按任意键退出...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()