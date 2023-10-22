import datetime
from ultralytics import YOLO
import cv2
import random
import os

#importing the DeepSort class from the deepsort_tracker module, and initializing the DeepSort object with the max_age parameter set to 50.
from deep_sort_realtime.deepsort_tracker import DeepSort

# from pygame import Color


# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
WHITE = (255,255,255)


#importing path to the video
path = os.path.join('.', 'data', 'trafficvid.mp4')

#location to saved video output
video_out = os.path.join('.', 'output_.mp4')

# initialize the video capture object
cap = cv2.VideoCapture(path)

#Getting frame width n height of the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# initialize the video writer object
#GETTING THE FRAMES PER SECOND(cv2.CAP_PROP_FPS),
#Video writer is used to define location for saved video output
writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'MP4V'),int(cap.get(cv2.CAP_PROP_FPS)),
                             (frame_width, frame_height))

# model_path = os.path.join('.', 'model','EyeSPEED/model/best.pt' ) -- >havent tested with this yet so do not uncomment

# load the custom YOLOv8n model with saved weights or use pretarained models
model = YOLO("yolov8n.pt")

#initisialize tracker
tracker = DeepSort(max_age=50)

#list containing 10 completely random colors
colors = [(random.randint(0,225),random.randint(0,225),random.randint(0,225)) for j in range(10)]

while True:
    #geting start time
    start = datetime.datetime.now()

    #reading frames from video
    ret, frame = cap.read()

    if not ret: #checks if there are any more frames left is none it breaks
        break

    # run the YOLO model on the frame
    results = model(frame)[0] #a list containing all the detections

    # initialize the list of bounding boxes and confidences
    results = []

    ######################################
    # DETECTION
    ######################################

    # loop over the detections  

     #unwrapping the list of detections
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
             #for testing(print)
            #  print(r) 
             x1,y1,x2,y2,score,class_id = r

             # extract the confidence (i.e., probability) associated with the prediction
             confidence = r[4]

             # filter out weak detections by ensuring the 
             # confidence is greater than the minimum confidence
             if float(confidence) < CONFIDENCE_THRESHOLD:
                continue
             # if the confidence is greater than the minimum confidence,
             # get the bounding box and the class id

             #Read them as integers
             x1 = int(x1)
             y1 = int(y1)
             x2 = int(x2)
             y2 = int(y2)
             class_id = int(class_id)

            #now we have unwrapped the detections, we need to call deepsort
             detections.append([x1,y1,x2-x1,y2-y1,confidence,class_id])

    

    ######################################
    # TRACKING
    ######################################

    # update the tracker with the new detections
    tracks = tracker.update_tracks(detections, frame=frame)
    # tracker.update(frame, detections)

    # loop over the tracks
    for track in tracks:
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue

        # get the track id and the bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        # draw the bounding box and the track id

        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)

        cv2.rectangle(frame, (x1, y1 - 20), (x1 + 20, y1), GREEN, -1)

        cv2.putText(frame, str(track_id), (x1 + 5, y1 - 8),cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        # cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
        #                             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"

    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    writer.write(frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
