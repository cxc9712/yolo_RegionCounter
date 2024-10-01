import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon


# Define queue areas 
#[bottom-left,bottom-right,top-right,top-left]
queue_areas = [

np.array([[0, 160], [640, 70], [640,0], [0, 0]]),
np.array([[0, 360], [640, 360], [640, 80], [0, 170]])
]

queue_colors = [
    (255, 0, 0),   
    (43, 128, 0),
    (255, 0, 255),  
    (0, 165, 255),  
    (128, 0, 128)   
]

def count_people_in_queues(frame,queue_areas):
    model = YOLO('yolov8n.pt')

    # Perform object detection
    results = model(frame,verbose=False)
    
    # Initialize counters for each queue
    queue_counts = [0] * len(queue_areas)
    
    # List to store bounding boxes, their colors, and additional info
    bounding_boxes = []
    
    # Process detections
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        for box, cls, conf in zip(boxes, classes, confidences):
            if cls == 0:  # Class 0 is person in COCO dataset
                x1, y1, x2, y2 = map(int, box)
                center_point = Point((x1 + x2) / 2, (y1 + y2) / 2)
                
                # Check if the person is in any of the queue areas
                in_queue = False
                queue_index = -1
                for i, queue_area in enumerate(queue_areas):
                    if Polygon(queue_area).contains(center_point):
                        queue_counts[i] += 1
                        in_queue = True
                        queue_index = i
                        break
                
                # Add bounding box with color based on which queue the person is in
                color = queue_colors[queue_index] if in_queue else (0, 0, 255)  # Queue color if in queue, red if not
                bounding_boxes.append((x1, y1, x2, y2, color, conf))
    
    return queue_counts, bounding_boxes

def run_bound_box(frame, queue_areas):
    # Open video capture (use 0 for webcam or provide video file path)
    cap = cv2.VideoCapture(frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get queue counts and bounding boxes
        counts, bounding_boxes = count_people_in_queues(frame,queue_areas)
        
        # Draw queue areas and display counts
        for i, (queue_area, count) in enumerate(zip(queue_areas, counts)):
            color = queue_colors[i % len(queue_colors)]
            cv2.polylines(frame, [queue_area], True, color, 2)
            cv2.putText(frame, f"Queue {i+1}: {count}", (queue_area[0][0], queue_area[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Draw bounding boxes with labels
        for x1, y1, x2, y2, color, conf in bounding_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label with class name and confidence
            label = f"Person: {conf:.2f}"
            
            # Get size of the text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw filled rectangle for text background
            cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
            
            # Put text on the filled rectangle
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow('Queue Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # cv2.destroyAllWindows()
            break
            

    cap.release()
    cv2.destroyAllWindows()
video_source = 'sample_video.mp4'
if __name__ == '__main__':
    run_bound_box(frame = video_source, queue_areas=queue_areas)

