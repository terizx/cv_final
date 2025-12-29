# Panorama Stitching with Advanced Blending 🖼️

## Project Overview
This project is a computer vision application developed for the **Final Project of Computer Vision**. 

It implements a robust image stitching pipeline capable of combining three overlapping images (Left, Middle, Right) into a single, seamless high-resolution panorama. The system is designed to handle hand-held shooting conditions, featuring **auto-exposure compensation** and **distance-weighted blending** to eliminate visible seams and lighting inconsistencies.

## ✨ Key Features

* **Multi-Image Stitching:** Seamlessly stitches 3 images based on a center-reference geometry.
* **Robust Feature Matching:** Utilizes **SIFT** (Scale-Invariant Feature Transform) with **RANSAC** for accurate homography estimation.
* **Exposure Compensation:** Automatically analyzes and adjusts brightness differences (Gain analysis in HSV space) between images to fix "day-and-night" inconsistencies.
* **Advanced Blending:** Implements **Distance-based Weighted Blending** (using Euclidean Distance Transform) to ensure smooth transitions and suppress ghosting caused by parallax.
* **Auto-Cropping:** Automatically removes black borders from the final stitched result.

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Libraries:**
    * `opencv-python` (Computer Vision algorithms)
    * `numpy` (Matrix operations)

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/terizx/cv_final.git](https://github.com/terizx/cv_final.git)
    cd cv_final
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy opencv-python
    ```

3.  **Prepare your images:**
    Place your 3 test images in the `images/` folder and name them:
    * `left.jpg`
    * `mid.jpg`
    * `right.jpg`

4.  **Run the script:**
    ```bash
    python stitch_3_imgs.py
    ```

5.  **Check the result:**
    The final panorama will be saved as `result.jpg` in the root directory.

## 📊 Methodology

The pipeline consists of 5 main stages:

1.  **Pre-processing:** Computes the brightness gain between overlapping images and applies exposure compensation.
2.  **Feature Matching:** Extracts SIFT keypoints and matches them using KNN (k=2) with Lowe's Ratio Test.
3.  **Homography:** Calculates the transformation matrix using RANSAC to align the Left and Right images to the Middle plane.
4.  **Warping & Blending:** Warps images to a common coordinate system and blends them using a **Distance Weight Map**, where pixels closer to the center have higher opacity.
5.  **Post-processing:** Thresholds the result to find the bounding box and crops out the black background.

## 🖼️ Results

| Input Sequence | Final Panorama |
| :---: | :---: |
| *(Place your input images here)* | ![Result](result.jpg) |

> *Note: The algorithm successfully handles exposure differences and slight parallax errors from hand-held shooting.*

## 👥 Team Members

* **[QIUZIXI]**: Algorithm implementation, Exposure Compensation, Blending logic.
* **[YANGZHIYI]**: Feature Matching pipeline, Testing, Project documentation.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.