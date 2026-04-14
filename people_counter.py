from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from itertools import zip_longest
from utils.mailer import Mailer
from imutils.video import FPS
from utils import thread
import numpy as np
import threading
import argparse
import datetime
import schedule
import logging
import imutils
import time
import dlib
import json
import csv
import cv2

# execution start time
start_time = time.time()

# setup logger
logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)

# initiate features config.
with open("utils/config.json", "r") as file:
    config = json.load(file)


def parse_arguments():
    # function to parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-p", "--prototxt", required=True,
        help="path to Caffe 'deploy' prototxt file"
    )
    ap.add_argument(
        "-m", "--model", required=True,
        help="path to Caffe pre-trained model"
    )
    ap.add_argument(
        "-i", "--input", type=str,
        help="path to optional input video file"
    )
    ap.add_argument(
        "-o", "--output", type=str,
        help="path to optional output video file"
    )
    ap.add_argument(
        "-c", "--confidence", type=float, default=0.4,
        help="minimum probability to filter weak detections"
    )
    ap.add_argument(
        "-s", "--skip-frames", type=int, default=30,
        help="# of skip frames between detections"
    )
    args = vars(ap.parse_args())
    return args


def send_mail():
    # function to send the email alerts
    Mailer().send(config["Email_Receive"])


def log_data(move_in, in_time, move_out, out_time):
    # function to log the counting data
    data = [move_in, in_time, move_out, out_time]
    export_data = zip_longest(*data, fillvalue='')

    with open('utils/data/logs/counting_data.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(("Move In", "In Time", "Move Out", "Out Time"))
        wr.writerows(export_data)


def read_frame(vs, use_threaded_stream=False):
    """
    Returns a frame from either:
    - cv2.VideoCapture -> (grabbed, frame)
    - custom threaded stream -> frame
    """
    if use_threaded_stream:
        frame = vs.read()
        if frame is None:
            return False, None
        return True, frame

    grabbed, frame = vs.read()
    if not grabbed:
        return False, None
    return True, frame


def people_counter():
    # main function for people_counter.py
    args = parse_arguments()

    # initialize the list of class labels MobileNet SSD was trained to detect
    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    ]

    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # decide input source
    use_threaded_stream = False

    if args.get("input"):
        print("[INFO] Starting the video..")
        vs = cv2.VideoCapture(args["input"])
    elif config.get("Thread"):
        print("[INFO] Starting the threaded stream..")
        stream_url = config.get("url", 0)
        vs = thread.ThreadingClass(stream_url)
        use_threaded_stream = True
    else:
        print("[INFO] Starting the live stream..")
        vs = cv2.VideoCapture(0)

    # initialize the video writer (we'll instantiate later if need be)
    writer = None

    # initialize the frame dimensions
    W = None
    H = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    # initialize the total number of frames processed thus far
    totalFrames = 0
    totalDown = 0
    totalUp = 0

    # initialize empty lists to store the counting data
    move_out = []
    move_in = []
    out_time = []
    in_time = []

    # alert cooldown so mails are not spammed every few frames
    last_alert_time = 0
    alert_cooldown_seconds = config.get("Alert_Cooldown", 60)

    # start the frames per second throughput estimator
    fps = FPS().start()

    # loop over frames from the video stream
    while True:
        grabbed, frame = read_frame(vs, use_threaded_stream=use_threaded_stream)

        # if we did not grab a frame then end stream/video
        if not grabbed or frame is None:
            break

        # resize the frame and convert the frame from BGR to RGB for dlib
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # if we are supposed to be writing a video to disk, initialize the writer
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

        # initialize the current status along with our list of bounding
        # box rectangles returned by either detector or trackers
        status = "Waiting"
        rects = []

        # run object detection every N frames
        if totalFrames % args["skip_frames"] == 0:
            status = "Detecting"
            trackers = []

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > args["confidence"]:
                    idx = int(detections[0, 0, i, 1])

                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)

                    rects.append((startX, startY, endX, endY))

        else:
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append((startX, startY, endX, endY))

        # draw a horizontal line in the center of the frame
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
        cv2.putText(
            frame,
            "-Prediction border - Entrance-",
            (10, H - 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

        # associate old object centroids with newly computed centroids
        objects = ct.update(rects)

        # compute current people count inside
        current_inside = totalDown - totalUp

        # loop over tracked objects
        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        move_out.append(totalUp)
                        out_time.append(date_time)
                        to.counted = True

                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        move_in.append(totalDown)
                        in_time.append(date_time)
                        to.counted = True

                    current_inside = totalDown - totalUp

                    if current_inside >= config["Threshold"]:
                        cv2.putText(
                            frame,
                            "-ALERT: People limit exceeded-",
                            (10, frame.shape[0] - 80),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.5,
                            (0, 0, 255),
                            2,
                        )

                        now = time.time()
                        if config["ALERT"] and now - last_alert_time >= alert_cooldown_seconds:
                            logger.info("Sending email alert..")
                            email_thread = threading.Thread(target=send_mail)
                            email_thread.daemon = True
                            email_thread.start()
                            logger.info("Alert sent!")
                            last_alert_time = now

            trackableObjects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(
                frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        # recalculate after all updates this frame
        current_inside = totalDown - totalUp

        if current_inside >= config["Threshold"]:
            cv2.putText(
                frame,
                "-ALERT: People limit exceeded-",
                (10, frame.shape[0] - 80),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        info_status = [
            ("Exit", totalUp),
            ("Enter", totalDown),
            ("Status", status),
        ]

        info_total = [
            ("Total people inside", current_inside),
        ]

        for (i, (k, v)) in enumerate(info_status):
            text = "{}: {}".format(k, v)
            cv2.putText(
                frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )

        for (i, (k, v)) in enumerate(info_total):
            text = "{}: {}".format(k, v)
            cv2.putText(
                frame, text, (265, H - ((i * 20) + 60)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        if config["Log"]:
            log_data(move_in, in_time, move_out, out_time)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        totalFrames += 1
        fps.update()

        if config["Timer"]:
            end_time = time.time()
            num_seconds = end_time - start_time
            if num_seconds > 28800:
                break

    fps.stop()
    logger.info("Elapsed time: {:.2f}".format(fps.elapsed()))
    logger.info("Approx. FPS: {:.2f}".format(fps.fps()))

    if writer is not None:
        writer.release()

    if use_threaded_stream:
        if hasattr(vs, "release"):
            vs.release()
        elif hasattr(vs, "stop"):
            vs.stop()
    else:
        vs.release()

    cv2.destroyAllWindows()


# initiate the scheduler
if config["Scheduler"]:
    # runs at every day (09:00 am)
    schedule.every().day.at("09:00").do(people_counter)
    while True:
        schedule.run_pending()
else:
    people_counter()
