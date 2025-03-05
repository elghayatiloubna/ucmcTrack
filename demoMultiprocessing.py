import cv2
import multiprocessing
import time
from ultralytics import YOLO
import json
from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper
import numpy as np

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


# Function to process a batch of frames
def process_frames(frame_list, thread_name, result_queue, process_id, cam_para_file, a1, a2, wx, wy, vmax, cdt, fps, high_score):
    start_time = time.time()
    
    # Load YOLO model and UCMCTrack tracker inside the process
    model = YOLO("pretrained/yolov8n.pt")
    mapper = Mapper(cam_para_file, "MOT17")
    tracker = UCMCTrack(a1, a2, wx, wy, vmax, cdt, fps, "MOT", high_score, False, None)
    
    # List to store detection data for the current batch of frames
    frame_data = []
    
    for i, frame in enumerate(frame_list):
        # Perform object detection
        results = model(frame)
        dets = []
        det_id = 0
        for result in results:
            for box in result.boxes:
                conf = box.conf.cpu().numpy()[0]
                bbox = box.xyxy.cpu().numpy()[0]  # bbox = [x1, y1, x2, y2]
                cls_id = box.cls.cpu().numpy()[0]
                w = bbox[2] - bbox[0]  # Width = x2 - x1
                h = bbox[3] - bbox[1]  # Height = y2 - y1
                
                # Filter out invalid or unwanted detections
                if w <= 10 and h <= 10 or cls_id not in [2, 5, 7] or conf <= 0.01:
                    continue
                
                # Create a Detection object
                det = Detection(det_id)
                det.bb_left = bbox[0]
                det.bb_top = bbox[1]
                det.bb_width = w
                det.bb_height = h
                det.conf = conf
                det.det_class = cls_id
                det.y, det.R = mapper.mapto([det.bb_left, det.bb_top, det.bb_width, det.bb_height])
                det_id += 1
                dets.append(det)
        
        # Update tracker with detections
        tracker.update(dets, i + 1)
        
        # Collect detection data for the current frame
        current_frame_data = {
            "frame_id": i + 1,
            "detections": []
        }
        for det in dets:
            if det.track_id > 0:
                detection_info = {
                    "track_id": det.track_id,
                    "bb_left": det.bb_left,
                    "bb_top": det.bb_top,
                    "bb_width": det.bb_width,
                    "bb_height": det.bb_height,
                    "confidence": det.conf,
                    "class": det.det_class,
                    "mapped_coordinates": [det.y[0, 0], det.y[1, 0]]
                }
                current_frame_data["detections"].append(detection_info)
        
        frame_data.append(current_frame_data)
        
        # Draw bounding boxes and IDs on the frame
        for det in dets:
            if det.track_id > 0:
                cv2.rectangle(frame, (int(det.bb_left), int(det.bb_top)), (int(det.bb_left + det.bb_width), int(det.bb_top + det.bb_height)), (0, 255, 0), 2)
                cv2.putText(frame, str(det.track_id), (int(det.bb_left), int(det.bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Put the processed frame and its data in the queue
        result_queue.put((process_id, i, frame, current_frame_data))
    
    print(f"[INFO] {thread_name} completed in {time.time() - start_time:.2f} sec")
    return frame_data

# Main function
def main():
    # Load video
    video_path = "C:\\Users\\lenovo\\Desktop\\deepLeaf\\video\\cars.mp44"
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Read all frames into a list
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"[INFO] Total frames: {len(frames)}")
    
    # Divide frames into `n` parts
    n = 4  # Number of processes (adjust based on CPU cores)
    frame_parts = [frames[i::n] for i in range(n)]
    
    # Create VideoWriter for output
    output_video_path = "C:\\Users\\lenovo\\Desktop\\deepLeaf\\UCMCTrack\\output\\output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Start timing
    start_total = time.time()
    
    # Create a queue for inter-process communication
    result_queue = multiprocessing.Queue()
    
    # Create and start processes
    processes = []
    for i in range(n):
        p = multiprocessing.Process(
            target=process_frames,
            args=(frame_parts[i], f"Process {i+1}", result_queue, i, "demo/cam_para.txt", 100.0, 100.0, 5, 5, 10, 10.0, fps, 0.5)
        )
        processes.append(p)
        p.start()
    
    # Create a dictionary to store processed frames and their data
    processed_frames_dict = {}
    processed_data_dict = {}
    total_frames_to_process = len(frames)
    
    # Collect results as they come in
    frames_processed = 0
    while frames_processed < total_frames_to_process:
        try:
            process_id, frame_idx, processed_frame, frame_data = result_queue.get(timeout=1.0)
            # Calculate the original position in the full frame list
            original_idx = process_id + frame_idx * n
            processed_frames_dict[original_idx] = processed_frame
            processed_data_dict[original_idx] = frame_data
            frames_processed += 1
            
            # Print progress
            if frames_processed % 10 == 0:
                print(f"[INFO] Processed {frames_processed}/{total_frames_to_process} frames")
                
        except queue.Empty:
            # Check if all processes are still alive
            if not any(p.is_alive() for p in processes):
                break
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Write output video in correct order
    for i in range(len(frames)):
        if i in processed_frames_dict:
            out.write(processed_frames_dict[i])
    
    out.release()
    
    # Combine detection data from all processes
    combined_frame_data = []
    for i in range(len(frames)):
        if i in processed_data_dict:
            combined_frame_data.append(processed_data_dict[i])
    
    # Save the combined detection data to a JSON file
    output_json_path = "./output/detections.json"
    with open(output_json_path, 'w') as json_file:
        json.dump(combined_frame_data, json_file, indent=4)
    
    print(f"[INFO] Total execution time: {time.time() - start_total:.2f} sec")
    print(f"[INFO] Detection data saved to {output_json_path}")
    print("[INFO] Video processing completed successfully.")

# Entry point
if __name__ == '__main__':
    import queue  # Import queue for the Queue.Empty exception
    multiprocessing.freeze_support()  # Required for Windows
    main()