# Traffic Monitoring and Analysis System

## Project Description
This project is an advanced image processing system that analyzes vehicle and pedestrian movements on video streams, detects violations and performs object counts. It is developed in Python using the OpenCV library.

## Key Features
- Customizable crosswalk and sidewalk zone definition
- Dynamic count lines for vehicle and pedestrian counting
- Vehicle violation detection in pedestrian zones
- Real-time object tracking and speed calculation

## Technical Features
- **Background Subtraction**: Dynamic object detection with KNN-based background subtraction
- **Morphological Operations**: Advanced filtering for noise reduction and object merging
- **Speed ​​Calculation**: Object motion analysis with pixel-based speed estimation
- **Multi-Region Support**: Manage multiple monitoring zones simultaneously

## Installation
1. Install requirements:
```bash
pip install opencv-python numpy
```

2. Download the project:
```bash
git clone https://github.com/Serkan0YLDZ/Traffic_Monitoring_and_Analysis_System.git
cd Traffic_Monitoring_and_Analysis_System
```

3. Run the project:
```bash
python main.py
```

## User Guide
1. **Initial Settings**
- Switch between modes with 'm' key:
- 🟢 Pedestrian Zone (Green)
- 🟣 Sidewalk Zone (Purple)
- 🟡 Counting Line (Yellow)
- Create zone/line with left click
- Start new zone with 'n' key
- Complete settings with 'q' key

![Screenshot from 2025-02-10 14-24-47](https://github.com/user-attachments/assets/7ca388b2-0e21-4221-82ac-68bbe4f88bf6)

2. **Real-time Monitoring**
- Tools shown in blue zones
- Pedestrians are followed in green zones
- Violations are highlighted in red
* If pedestrians walk in the roadway or
* If vehicles do not wait for people walking in the roadway, they are labeled red
- Vehicles and pedestrians crossing the yellow line are listed in the top left corner

![Screenshot from 2025-02-10 14-25-19](https://github.com/user-attachments/assets/d91525f5-b8e2-4289-9a60-9f0dbd7d818e)