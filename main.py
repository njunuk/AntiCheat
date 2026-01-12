import cv2
from ultralytics import YOLO
from tkinter import *
from datetime import datetime
import time
import os
import threading
import tempfile, base64, zlib

WRIST_T = 0.85  # Wrist confidence threshold

LEFT_WRIST = 9 # COCO keypoint indexes
RIGHT_WRIST = 10

# Hand hidden logic
def hand_hidden_logic(kps, conf):
    lw = conf[LEFT_WRIST]  if conf is not None else 0
    rw = conf[RIGHT_WRIST] if conf is not None else 0

    left_hidden  = lw < WRIST_T
    right_hidden = rw < WRIST_T

    return left_hidden, right_hidden

def start():
    # Settings
    MODEL = "yolov8" + em.get() + ".pt"
    MODEL2 = "yolov8" + em.get() + "-pose.pt"
    CAMERA = int(ev.get()) # Camera index
    OUTPUT_DIR = "logs"
    HAND_DETECTION_ENABLED = bool(hdcbv.get()) # Hand detection switch
    CONF = float(dc.get())  # Detection confidence
    COOLDOWN_SECONDS = int(cds.get()) # Between snapshots in general
    FRAMES_REQUIRED = int(frp.get()) # For phone to snapshot
    SECONDS_REQUIRED = int(srh.get()) # For hand to snapshot

    # Other
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    last_save_time = 0.0
    consecutive_frames = 0
    hand_frames = 0

    # YOLO model
    model = YOLO(MODEL)
    if HAND_DETECTION_ENABLED == True:
        model2 = YOLO(MODEL2)
    cvv = cv2.VideoCapture(CAMERA)

    prev_time = time.time()

    while True:
        ret, frame = cvv.read()
        if not ret:
            break

        results = model(frame, conf=CONF, imgsz=640, verbose=False)
        if HAND_DETECTION_ENABLED == True:
            results2 = model2(frame, conf=CONF, imgsz=640, verbose=False)
        phone_detected = False
        hand_detected = False

        # Fps counter
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        fps_int = int(fps)
        prev_time = current_time
        cv2.putText(
            frame,
            f"FPS: {fps_int}",
            (5, 20),  # top-left corner
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # green text
            1,
            cv2.LINE_AA
        )
        # Fps counter end
        # -------------------------------------------
        # YOLO PHONE DETECTION
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if model.names[cls] == "cell phone":
                    phone_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(frame, "Phone", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # -------------------------------------------

        # -------------------------------------------
        # YOLO HAND DETECTION
        if HAND_DETECTION_ENABLED == True:
            rr = results2[0]
            boxes = rr.boxes.xyxy.cpu().numpy() if rr.boxes is not None else []
            keypoints = rr.keypoints.xy.cpu().numpy() if hasattr(rr.keypoints, "xy") else []
            keyconfs = rr.keypoints.conf.cpu().numpy() if hasattr(rr.keypoints, "conf") else None

            # For each person:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)

                kps = keypoints[i]
                confs = keyconfs[i] if keyconfs is not None else None

                # Hand hidden logic
                left_hidden, right_hidden = hand_hidden_logic(kps, confs)

                # Decide box color
                if left_hidden or right_hidden:
                    color = (0, 0, 255)  # Red
                    hand_detected = True
                    cv2.putText(frame, "Hands Hidden", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    color = (0, 255, 0)  # Green
                    hand_detected = False

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # -------------------------------------------

        # -------------------------------------------
        # SNAPSHOT LOGIC
        now = time.time()
        # Phone caused snapshot logic
        if phone_detected:
            consecutive_frames += 1
        else:
            consecutive_frames = 0

        if consecutive_frames >= FRAMES_REQUIRED and (now - last_save_time) >= COOLDOWN_SECONDS:
            filename = f"detection_phone_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(filepath, frame)
            last_save_time = now
            consecutive_frames = 0
        # Hidden hand caused snapshot logic
        if hand_detected:
            hand_frames += 1
        else:
            hand_frames = 0

        if hand_frames >= fps_int*SECONDS_REQUIRED and (now - last_save_time) >= COOLDOWN_SECONDS:
            filename = f"detection_hand_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(filepath, frame)
            last_save_time = now
            hand_frames = 0
        # -------------------------------------------
        cv2.imshow("Phone Detection", frame) # Start YOLO detection

        if cv2.waitKey(1) & 0xFF == ord("q"): # Close YOLO detection hotkey
            break

    cvv.release()
    cv2.destroyAllWindows()

def ENGTranslate():
    Lab1.config(text="Choose YOLO model (n/s/m/l/x)")
    Lab2.config(text="Enter camera index (0/1/2)")
    Lab3.config(text="Detection confidence required")
    Lab4.config(text="Cooldown seconds between snapshots")
    Lab5.config(text="Frames required for phone snapshot")
    Lab6.config(text="Seconds required for hidden hands snapshot")
    hdcb.config(text="Hand detection")
    StrtBtn.config(text="Start")

def UATranslate():
    Lab1.config(text="Виберіть модель YOLO (n/s/m/l/x)")
    Lab2.config(text="Введіть індекс камери (0/1/2)")
    Lab3.config(text="Мінімальна впевненість детекції")
    Lab4.config(text="Затримка між знімками (сек)")
    Lab5.config(text="Кількість кадрів для знімка телефону")
    Lab6.config(text="Кількість Секунд для знімка прихованих рук")
    hdcb.config(text="Детекція рук")
    StrtBtn.config(text="Старт")


# -------------------------------------------------------
# TKINTER UI
root = Tk()
# Title and transparent Icon
root.title("Anti-cheat")
root.iconbitmap("favicon.ico")
models = ["n", "s", "m", "l", "x"]
em = StringVar(value="n")

Lab1 =Label(root, text="Choose YOLO model (n/s/m/l/x)")
Lab1.grid(row=0, column=0, padx=5)
OptionMenu(root, em, *models).grid(row=0, column=1, padx=5)

Lab2 =Label(root, text="Enter camera index (0/1/2)")
Lab2.grid(row=1, column=0, padx=5)
evvar = IntVar()
evvar.set(0)
ev = Entry(root, textvariable=evvar)
ev.grid(row=1, column=1, padx=5)

Lab3 =Label(root, text="Detection confidence required")
Lab3.grid(row=2, column=0, padx=5)
dcvar = DoubleVar()
dcvar.set(0.35)
dc = Entry(root, textvariable=dcvar)
dc.grid(row=2, column=1, padx=5)

Lab4 =Label(root, text="Cooldown seconds between snapshots")
Lab4.grid(row=3, column=0, padx=5)
cdsvar = IntVar()
cdsvar.set(5)
cds = Entry(root, textvariable=cdsvar)
cds.grid(row=3, column=1, padx=5)

Lab5 =Label(root, text="Frames required for phone snapshot")
Lab5.grid(row=4, column=0, padx=5)
frpvar = IntVar()
frpvar.set(3)
frp = Entry(root, textvariable=frpvar)
frp.grid(row=4, column=1, padx=5)

Lab6 =Label(root, text="Seconds required for hidden hands snapshot")
Lab6.grid(row=5, column=0, padx=5)
srhvar = IntVar()
srhvar.set(3)
srh = Entry(root, textvariable=srhvar)
srh.grid(row=5, column=1, padx=5)

hdcbv = BooleanVar()
hdcbv.set(True)
hdcb = Checkbutton(root, text="Hand detection", variable=hdcbv)
hdcb.grid(row=6, column=0, padx=5)

ButtonFrame = Frame(root)
ButtonFrame.grid(row=7, column=1, padx=5)

StrtBtn = Button(root, text="Start", command=lambda: threading.Thread(target=start).start())
StrtBtn.grid(row=7, column=0, padx=5)
Button(ButtonFrame, text="UA", command=lambda: threading.Thread(target=UATranslate).start()).grid(row=0, column=0, padx=5)
Button(ButtonFrame, text="ENG", command=lambda: threading.Thread(target=ENGTranslate).start()).grid(row=0, column=1, padx=5)
# -------------------------------------------------------

root.mainloop()
