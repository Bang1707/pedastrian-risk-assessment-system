# Pedestrian Risk Assessment System (PRAS)

A vision-based pedestrian collision risk estimation system built using object detection, custom tracking, and time-to-collision (TTC) modeling.

## 📌 Project Overview

This project aims to estimate pedestrian collision risk from video input using:

- Pretrained object detection (YOLO)
- Custom-built Kalman Filter tracker
- Motion estimation
- Time-to-Collision (TTC) calculation
- Heuristic-based risk scoring

The main contribution of the project lies in the tracking and risk modeling layers rather than object detection.

## 🧠 System Pipeline

Video Input  
→ Object Detection  
→ Multi-Object Tracking  
→ Motion Estimation  
→ TTC Calculation  
→ Risk Scoring  
→ Visualization & Analytics  

## 📂 Project Structure
pedestrian-risk-assessment/
│
├── data/
├── models/
├── src/
│ ├── detection/
│ ├── tracking/
│ ├── motion/
│ ├── risk/
│ ├── visualization/
│ └── utils/
│
├── analytics/
├── experiments/
├── config/
├── main.py
└── requirements.txt


## 🚀 Goals

- Develop a mathematically explainable tracking model
- Implement a custom Kalman filter tracker
- Design a heuristic collision risk model
- Evaluate system performance on real-world traffic footage

## ⚙️ Technologies

- Python
- OpenCV
- YOLO (Ultralytics)
- NumPy
- SciPy


Author: Mesut Deniz Zeka
