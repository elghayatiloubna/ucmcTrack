from ultralytics import YOLO
import os,cv2
import argparse

from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper
import numpy as np
import json

# 定义一个Detection类，包含id,bb_left,bb_top,bb_width,bb_height,conf,det_class
class Detection:
    """
        Initializes a Detection object with the following attributes:
        - id: Unique identifier for the detection.
        - bb_left: X-coordinate of the top-left corner of the bounding box.
        - bb_top: Y-coordinate of the top-left corner of the bounding box.
        - bb_width: Width of the bounding box.
        - bb_height: Height of the bounding box.
        - conf: Confidence score of the detection (how likely it is to be correct).
        - det_class: Class ID of the detected object (e.g., 0 for person, 1 for car, etc.).
        - track_id: ID assigned by the tracker (default is 0, meaning not yet tracked).
        - y: A 2x1 numpy array representing the mapped coordinates of the detection (default is zeros).
        - R: A 4x4 numpy identity matrix representing uncertainty or covariance (default is identity matrix).
    """
    def __init__(self, id, bb_left = 0, bb_top = 0, bb_width = 0, bb_height = 0, conf = 0, det_class = 0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)


    def __str__(self):
        """
        Returns a string representation of the Detection object.
        This is used for printing the object in a human-readable format.
        output is : d1, bb_box:[100,150,50,60], conf=0.95, class2, uv:[125,210], mapped to:[10.5,20.3]
        d1: Detection ID.
        bb_box:[100,150,50,60]: Bounding box in image coordinates (pixels).
        conf=0.95: Confidence score.
        class2: Class ID of the detected object.
        uv:[125,210]: Center of the bounding box in image coordinates.
        mapped to:[10.5,20.3]: Mapped coordinates in another coordinate system (e.g., world coordinates).
        """
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left+self.bb_width/2,self.bb_top+self.bb_height,self.y[0,0],self.y[1,0])

    def __repr__(self):
        return self.__str__()


# Detector类，用于从Yolo检测器获取目标检测的结果
class Detector:
    def __init__(self):
        self.seq_length = 0 #notUsed
        self.gmc = None #notUsed

    def load(self,cam_para_file):
        self.mapper = Mapper(cam_para_file,"MOT17")
        self.model = YOLO('pretrained/yolov8n.pt')

    def get_dets(self, img,conf_thresh = 0,det_classes = [0]):
        #Only detections with a confidence score greater than or equal to conf_thresh will be considered valid and included in the results.
        
        dets = [] #This list is used to store all the detection objects
          
        # Convert the frame from BGR to RGB (since OpenCV uses BGR format) 
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

        # 使用 RTDETR 进行推理  
        results = self.model(frame,imgsz = 1088)
        #the output of numpy is always an array
        # .cpu() is a PyTorch method that moves the tensor from the GPU to the CPU, since numpy cannot handle GPU tensors.
        det_id = 0
        for box in results[0].boxes:
            conf = box.conf.cpu().numpy()[0]
            bbox = box.xyxy.cpu().numpy()[0] # bbox = [x1, y1, x2, y2]
            cls_id  = box.cls.cpu().numpy()[0] 
            w = bbox[2] - bbox[0] # Width = x2 - x1
            h = bbox[3] - bbox[1] # Height = y2 - y1
            # Filter out invalid or unwanted detections
            if w <= 10 and h <= 10 or cls_id not in det_classes or conf <= conf_thresh:
                continue

            # 新建一个Detection对象
            det = Detection(det_id) # Create a new Detection object
            det.bb_left = bbox[0] #x1
            det.bb_top = bbox[1] #y1
            det.bb_width = w
            det.bb_height = h
            det.conf = conf
            det.det_class = cls_id
            det.y,det.R = self.mapper.mapto([det.bb_left,det.bb_top,det.bb_width,det.bb_height]) # see mapper.py
            det_id += 1

            dets.append(det)

        return dets
    

import json

def main(args):
    class_list = [2, 5, 7]

    cap = cv2.VideoCapture(args.video)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_out = cv2.VideoWriter("C:\\Users\\lenovo\\Desktop\\deepLeaf\\UCMCTrack\\output\\output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))  

    # Create a resizable OpenCV window
    cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("demo", width, height)

    detector = Detector()
    detector.load(args.cam_para)

    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score, False, None)

    # To store detected objects for each frame
    frame_data = []

    # Loop through video frames
    frame_id = 1
    while True:
        ret, frame_img = cap.read()
        if not ret:  
            break
    
        dets = detector.get_dets(frame_img, args.conf_thresh, class_list)
        tracker.update(dets, frame_id)

        # Dictionary to store detected objects in the current frame
        current_frame_data = {
            "frame_id": frame_id,
            "detections": []
        }

        for det in dets:
            # If the object is being tracked
            if det.track_id > 0:
                # Add detection data to the current frame's dictionary
                detection_info = {
                    "track_id": det.track_id,
                    "bb_left": det.bb_left,
                    "bb_top": det.bb_top,
                    "bb_width": det.bb_width,
                    "bb_height": det.bb_height,
                    "confidence": det.conf,
                    "class": det.det_class,
                    "mapped_coordinates": [det.y[0, 0], det.y[1, 0]]  # Mapped coordinates
                }
                current_frame_data["detections"].append(detection_info)

                # Draw bounding box and ID on the frame
                cv2.rectangle(frame_img, (int(det.bb_left), int(det.bb_top)), (int(det.bb_left + det.bb_width), int(det.bb_top + det.bb_height)), (0, 255, 0), 2)
                cv2.putText(frame_img, str(det.track_id), (int(det.bb_left), int(det.bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Append the current frame's data to the list
        frame_data.append(current_frame_data)

        frame_id += 1

        # Display the current frame
        cv2.imshow("demo", frame_img)
        cv2.waitKey(1)

        # Write the frame to the output video
        video_out.write(frame_img)
    
    # Release video capture and writer
    cap.release()
    video_out.release()
    cv2.destroyAllWindows()
                          # Helper function to convert NumPy types to native Python types
    def convert_to_python_types(obj):
        if isinstance(obj, np.generic):
            return obj.item()  # Convert NumPy scalar to Python scalar
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to Python list
        elif isinstance(obj, dict):
            return {key: convert_to_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        else:
            return obj
        
    # Save the frame data to a JSON file
    output_json_path = "C:\\Users\\lenovo\\Desktop\\deepLeaf\\UCMCTrack\\output\\detections.json"
    with open(output_json_path, 'w') as json_file:
        json.dump(frame_data, json_file, indent=4)
    # Convert all NumPy types to native Python types before saving
    frame_data_converted = convert_to_python_types(frame_data)
    json.dump(frame_data_converted, json_file, indent=4)
    print(f"Detection data saved to {output_json_path}")


parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--video', type=str, default = "C:\\Users\\lenovo\\Desktop\\deepLeaf\\video\\cars.mp4", help='video file name')
parser.add_argument('--cam_para', type=str, default = "demo/cam_para.txt", help='camera parameter file name')
parser.add_argument('--wx', type=float, default=5, help='wx')
parser.add_argument('--wy', type=float, default=5, help='wy')
parser.add_argument('--vmax', type=float, default=10, help='vmax')
parser.add_argument('--a', type=float, default=100.0, help='assignment threshold')
parser.add_argument('--cdt', type=float, default=10.0, help='coasted deletion time')
parser.add_argument('--high_score', type=float, default=0.5, help='high score threshold')
parser.add_argument('--conf_thresh', type=float, default=0.01, help='detection confidence threshold')
args = parser.parse_args()

main(args)
