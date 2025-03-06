# Sawlog branch whorl detector for terrestrial laser scanning point clouds

**1.** Renders images of sawlog partitions from point clouds in four directions.

**2.** Uses pre-trained Yolov5 model (whorlDetector.pt) to detect branch whorls from TLSimages.

**3.** Combines the detections of the different direction by linearly clustering over-lapping detections.

**4.** Estimates sawlog partitions branch whorl count (whorl count) ja mean distance between branch whorls (mean whorl distance).

whorlDetect.ipynb contains code in Jupyter notebook format.
First code block installs depencies. 
Second block runs whorl detector looping through tree_log_locations.xlsx table where is locations of to be rendered trees and sawlogs partitions (x and y of tree. Heights of the sawlog).
Detector uses point clouds of trees from point_clouds -folder.
Finally detector saves the estimated values into whorls.csv.

