import cv2
import multiprocessing
import time
from ultralytics import YOLO

# Function to process a batch of frames
def process_frames(frame_list, thread_name, result_queue, process_id):
    start_time = time.time()
    #processed_frames = []
   
    # Load YOLO model inside the process
    model = YOLO("./yolov8n.pt")
   
    for i, frame in enumerate(frame_list):
        results = model(frame)
        # Draw detections
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{model.names[int(box.cls[0])]} {box.conf[0]:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
       
        # Put the frame in the queue with its original position information
        result_queue.put((process_id, i, frame))
    
    print(f"[INFO] {thread_name} completed in {time.time() - start_time:.2f} sec")

# Main function
def main():
    # Load video
    video_path = "./traffic.mp4"
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
    output_video_path = "./output/output.mp4"
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
            args=(frame_parts[i], f"Process {i+1}", result_queue, i)
        )
        processes.append(p)
        p.start()
    
    # Create a dictionary to store processed frames with their position
    processed_frames_dict = {}
    total_frames_to_process = len(frames)
    
    # Collect results as they come in
    frames_processed = 0
    while frames_processed < total_frames_to_process:
        try:
            process_id, frame_idx, processed_frame = result_queue.get(timeout=1.0)
            # Calculate the original position in the full frame list
            original_idx = process_id + frame_idx * n
            processed_frames_dict[original_idx] = processed_frame
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
    print(f"[INFO] Total execution time: {time.time() - start_total:.2f} sec")
    print("[INFO] Video processing completed successfully.")

# Entry point
if __name__ == '__main__':
    import queue  # Import queue for the Queue.Empty exception
    multiprocessing.freeze_support()  # Required for Windows
    main()