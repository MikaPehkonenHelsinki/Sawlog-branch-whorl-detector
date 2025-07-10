## import dependencies
import laspy
import pandas as pd
import numpy as np
from statistics import mean
import os
from PIL import Image
import subprocess

script_dir = os.getcwd() # set working directory (default ipynb location)
os.chdir(script_dir)
print("Current folder:" + os.getcwd())

linearClusteringTreshold = 0.09 # set linear clustering threshold (cm)
TreePointCloudFolder = script_dir+"/point_clouds/" # tree point cloud locations
treeLocations = pd.read_excel(
    os.path.join(script_dir,'tree_log_locations.xlsx'), skiprows=0) # locations of the trees and logs
model = os.path.join(script_dir, "whorlDetector.pt") # yolov5 model location
TLSimages_parent = os.path.join(script_dir, "TLSimages") # to be rendered TLS images
whorl_df = pd.DataFrame(columns=['tree', 'log', 'whorl_count' , 'mean_whorl_distance']) #final whorl dataframe

for index, row in treeLocations.iterrows():
    import open3d as o3d
    # Get log specs
    tree_id = str(int(row['tree']))
    log_id = str(int(row['log']))
    logBottomHeight = row['log_bottom_h'] 
    logTopHeight = row['log_top_h'] 
    logLength = logTopHeight - logBottomHeight
    stemLocationX = row['tree_X'] 
    stemLocationY = row['tree_Y'] 
    TreePointCloud = TreePointCloudFolder + tree_id + '.las'
    if not os.path.exists(TreePointCloud):
        print(f"File {TreePointCloud} not found. Stopping loop.")
        break
    # ---- MAKE TLSIMAGES FOLDER FOR THIS LOG ONLY ----
    TLSimages = f"{TLSimages_parent}/{tree_id}_{log_id}"
    os.makedirs(TLSimages, exist_ok=True)
    # Read point cloud
    las = laspy.read(TreePointCloud)
    new_file = laspy.create(point_format=las.header.point_format,
        file_version=las.header.version)
    new_file.points = las.points
    las = new_file
    # Cut the sawlog from point cloud
    new_file.points = new_file.points[las.z >= (logBottomHeight)] 
    new_file.points = new_file.points[las.z <= (logTopHeight)]
    # Get XY middle point of the point cloud
    middleX = stemLocationX
    middleY = stemLocationY
    # Cut points 0.5 meter away from middle
    middle_dist = 1 # original was 0.5
    new_file.points = las.points[las.x <= middleX+middle_dist]
    new_file.points = las.points[las.x >= middleX-middle_dist]
    new_file.points = las.points[las.y <= middleY+middle_dist]
    new_file.points = las.points[las.y >= middleY-middle_dist]
    las = new_file
    print(f"Number of points: {len(las.points)}")
    # create open3d cloud
    point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(point_data)
    geom.paint_uniform_color([1, 1, 1])
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name='Open3D', width=1000*2, height=1000*3, left=0, top=0, visible=True)
    visualizer.add_geometry(geom)
    render = visualizer.get_render_option()
    render.background_color = np.array([0, 0, 0])
    render.point_size = 5
    view_ctl = visualizer.get_view_control()
    view_ctl.change_field_of_view(-60)
    view_ctl.set_zoom(0.55)
    # 1. dir image capture
    view_ctl.set_up((1, 0, 0))  
    view_ctl.set_up((0, 0, 1)) 
    view_ctl.set_front((0, 1, 0)) 
    visualizer.capture_screen_image(TLSimages+"/"+tree_id+'_'+log_id+"_1.png", do_render=True)
    # 2. dir image capture
    view_ctl.set_up((1, 0, 0))  
    view_ctl.set_up((0, 0, 1))  
    view_ctl.set_front((-1, 0, 0))  
    visualizer.capture_screen_image(TLSimages+"/"+tree_id+'_'+log_id+"_2.png", do_render=True)
    # 3. dir image capture
    view_ctl.set_up((1, 0, 0)) 
    view_ctl.set_up((0, 0, 1))  
    view_ctl.set_front((0, -1, 0))  
    visualizer.capture_screen_image(TLSimages+"/"+tree_id+'_'+log_id+"_3.png", do_render=True)
    # 4. dir image capture
    view_ctl.set_up((1, 0, 0))  
    view_ctl.set_up((0, 0, 1))  
    view_ctl.set_front((1, 0, 0))  
    visualizer.capture_screen_image(TLSimages+"/"+tree_id+'_'+log_id+"_4.png", do_render=True)

    # Crop black bars of the images
    folder_path = TLSimages
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image_array = np.array(image)
            non_black_pixels = np.any(image_array != [0, 0, 0], axis=-1)
            rows = np.any(non_black_pixels, axis=1)
            cols = np.any(non_black_pixels, axis=0)
            top, bottom = np.argmax(rows), len(rows) - np.argmax(rows[::-1]) - 1
            left, right = np.argmax(cols), len(cols) - np.argmax(cols[::-1]) - 1
            cropped_image = image.crop((left, top, right, bottom))
            cropped_image.save(os.path.join(folder_path, f"{filename}"))

    # Destroy open3d objects
    del o3d, visualizer, view_ctl, render

    # ---- SAVING DETECTIONS INSTEAD OF ALWAYS DELETING ----
    detections = os.path.join(TLSimages, "detections")
    os.makedirs(detections, exist_ok=True)
    
    # ---- REPLACING MAGIC COMMANDS ----
    # Change to yolov5 directory, run detection, change back
    yolov5_dir = os.path.join(script_dir, "yolov5")
    os.chdir(yolov5_dir)
    # Use subprocess to call YOLO detection
    subprocess.run([
        "python", "detect.py",
        "--source", TLSimages,
        "--weights", model,
        "--project", detections,
        "--name", "run",
        "--save-txt",
        "--conf", "0.30"
    ])
    os.chdir(script_dir)
    # ---- END REPLACE ----

    # detection label folder files
    labelDirection = os.path.join(detections, "run", "labels")
    label_files = os.listdir(labelDirection)
    whorls=[]

    for filename in os.listdir(labelDirection):
        label = os.path.join(labelDirection, filename)
        detectionLabels = pd.read_csv(label, sep=' ', engine='python', header=None)
        detectionsDeNormalized = ((1-detectionLabels[2]) * (logLength)) + logBottomHeight
        whorls += detectionsDeNormalized.tolist()
    # ---- CHECK THAT THERE ARE ACUTALLY SOME WHORLS TO NOT BREAK  ----
    if len(whorls) > 1:
        whorls.sort()
        n = linearClusteringTreshold
        nnumbers = np.array(whorls)
        clusters = pd.DataFrame({
            'numbers': whorls,
            'segment': np.cumsum([0] + list(1*(nnumbers[1:] - nnumbers[0:-1] > n))) + 1
        }).groupby('segment').agg({'numbers': set}).to_dict()['numbers']
        klust = list()
        for x in range(len(clusters)):
            klust.append(mean(clusters[x+1]))
        whorls = klust
        whorls.sort()

    whorlCount = len(whorls)
    meanDist = []
    if len(whorls) > 1:
        for x in range(len(whorls)-1):
            meanDist.append(whorls[x+1]-whorls[x])
        meanWhorlDistance = mean(meanDist)*100
    else:
        meanWhorlDistance = -1

    print("Number of whorls: " + str(whorlCount))
    print("Mean distance between whorls: " + str(meanWhorlDistance) + " cm")

    whorl_df.loc[len(whorl_df)] = [tree_id, log_id, whorlCount, meanWhorlDistance]

print("Job is done!")
whorl_df.to_csv(os.path.join(script_dir, "whorls.csv"), index=False)