# DLR-bridge-segmentation
Final Research Project at German Aerospace Center(DLR) commmunications and navigation internship, researching bridge segmentation and river bank detection from lidar point clouds

Before and after segmentation process, output composing point clouds of individual bridges and river banks
![Bridge](https://github.com/0austinli4/DLR-bridge-segmentation/blob/main/bridgeSegmentation%201.gif)

### Problem Context
Given a point cloud generated from lidar scans and composed from simulataneous localization and mapping, detect regions of interest. 

I developed an algorithm to perform segmentatino of the point clouds, following three steps: pre-processing, clustering, and geometric segmentation. Using existing algorithms like DBScan and RANSAC Plane Segmentation, this project creates a way detect bridge components from inland waterway point cloud scans. 

### Applications
Potential applications include: contour formation / bridge structure assessment, autonomous vesssel navigation, semantic segmentation

## Demo

Individual Bridge Detection
![Single Bridge](https://github.com/0austinli4/DLR-bridge-segmentation/blob/main/singleBridge.gif)

<img width="800" alt="image" src="https://github.com/0austinli4/DLR-bridge-segmentation/assets/108751584/1381686b-8d82-40ab-ae2e-eee0785d9193">

### Full Method Video
Full method demonstration video: https://drive.google.com/file/d/1JrH6ahYkYQm1fiwVMAyj4BqhdDnfBXpG/view?resourcekey

### Presentation
https://docs.google.com/presentation/d/1s5RY8UdkkgSWvr9eEG0s9PM1ylu1SxAu/edit?usp=sharing&ouid=112093100325586153117&rtpof=true&sd=true
