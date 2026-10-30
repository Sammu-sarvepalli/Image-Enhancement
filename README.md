💻B.Tech Final Year Project (CSE)

# 💡Title:
Efficient and Effective Blind JPEG Image Improvement with Sequential Feature Processing

# 📄 Abstract 
Image compression like JPEG degrades quality, causing blocking & mosquito noise artifacts. This paper proposes a method with a **Quality Estimation Part (QE Part)** that estimates input quality and **Image Processing Parts (IP Parts)** in cascade for sequential processing. This achieves state-of-the-art quantitative results (PSNR/SSIM) with smaller computational resources.

# 🎯 Problem Statement
JPEG compression = efficient storage but poor visual quality due to 8x8 blocking artifacts. Goal is to restore high-quality images blindly (without knowing Quality Factor).

This project aims to enhance low-quality, compressed JPEG images without prior knowledge of their compression level (Blind Enhancement) using Sequential Feature Processing and Convolutional Neural Networks (CNN).

# 🛠️ Tech Stack
Language: Python
Libraries: OpenCV, NumPy, Matplotlib, Scikit-Image, PyQt
Deep Learning: CNN (Sequential Feature Processing)
Concepts: Image Restoration, Feature Extraction, Artifact Removal

# ✨ Features & Methodology
Blind Enhancement: Works without knowing JPEG Quality Factor.
Sequential Feature Processing: Extracts features step-by-step for better restoration.
Artifact Removal: Effectively removes blocking artifacts.

# 💭 Proposed System
1.  **QE Part:** Estimates the quality representation of the input JPEG.
2.  **IP Parts Cascade:** Multiple Processing Blocks connected sequentially to process different features efficiently.
3.  **Blind Enhancement:** Works for JPEGs with various qualities.

# 📁 Repository Structure
/Documentation - Full Project Report PDF
/Code - Implementation (QE + IP Parts)
/Results - Input vs Enhanced Output Images
/UML - Use-case, Class, Sequence, Activity Diagrams

# 🧑‍💻 How to Run
pip install opencv-python numpy matplotlib pillow👩‍💻 

# 🚀 Results
- Achieves higher PSNR & SSIM than conventional methods.
- Effective for large-size images with less computing cost.
- Better qualitative visual results.

# 🧑‍💻Author
Samyuktha Sarvepalli
B.Tech - Computer Science & Engineering

# 📜 Reference
Final Year Project
Narayana Engineering College, Gudur - Affiliated to JNTUA.
