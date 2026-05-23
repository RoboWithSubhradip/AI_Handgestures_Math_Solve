import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import easyocr

# ============================================
# CAMERA SETUP
# ============================================
cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)

# ============================================
# HAND DETECTOR
# ============================================
detector = HandDetector(
    detectionCon=0.8,
    maxHands=1
)

# ============================================
# OCR READER
# ============================================
reader = easyocr.Reader(['en'])

# ============================================
# CREATE CANVAS
# ============================================
canvas = np.zeros((720, 1280, 3), np.uint8)

# ============================================
# VARIABLES
# ============================================
xp, yp = 0, 0

brushThickness = 15

answer = ""

last_solve_time = 0

# ============================================
# MAIN LOOP
# ============================================
while True:

    success, img = cap.read()

    if not success:
        break

    # FLIP IMAGE
    img = cv2.flip(img, 1)

    # ============================================
    # HAND DETECTION
    # ============================================
    hands, img = detector.findHands(img)

    if hands:

        hand = hands[0]

        lmList = hand["lmList"]

        fingers = detector.fingersUp(hand)

        # INDEX FINGER TIP
        x1, y1 = lmList[8][0], lmList[8][1]

        # MIDDLE FINGER TIP
        x2, y2 = lmList[12][0], lmList[12][1]

        # ============================================
        # DRAW MODE
        # PINCH INDEX + MIDDLE
        # ============================================
        distance, info, img = detector.findDistance(
            lmList[8][0:2],
            lmList[12][0:2],
            img
        )

        if distance < 40:

            cv2.putText(
                img,
                "DRAW MODE",
                (980, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(
                canvas,
                (xp, yp),
                (x1, y1),
                (255, 255, 255),
                brushThickness
            )

            xp, yp = x1, y1

        # ============================================
        # MOVE MODE
        # INDEX + MIDDLE OPEN
        # ============================================
        elif fingers[1] == 1 and fingers[2] == 1:

            xp, yp = 0, 0

            cv2.putText(
                img,
                "MOVE MODE",
                (980, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2
            )

        # ============================================
        # CLEAR MODE
        # ALL FINGERS OPEN
        # ============================================
        if fingers == [1, 1, 1, 1, 1]:

            canvas = np.zeros((720, 1280, 3), np.uint8)

            answer = ""

            xp, yp = 0, 0

            cv2.putText(
                img,
                "CLEAR",
                (1000, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ============================================
        # SOLVE MODE
        # THUMB UP ONLY
        # ============================================
        thumb_up = (
            fingers[0] == 1 and
            fingers[1] == 0 and
            fingers[2] == 0 and
            fingers[3] == 0 and
            fingers[4] == 0
        )

        if thumb_up:

            cv2.putText(
                img,
                "SOLVING...",
                (930, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            current_time = cv2.getTickCount() / cv2.getTickFrequency()

            # PREVENT MULTIPLE SOLVE
            if current_time - last_solve_time > 2:

                last_solve_time = current_time

                # ============================================
                # OCR PROCESS
                # ============================================
                gray = cv2.cvtColor(
                    canvas,
                    cv2.COLOR_BGR2GRAY
                )

                results = reader.readtext(gray)

                detected_text = ""

                for r in results:
                    detected_text += r[1]

                # FIX COMMON OCR ERRORS
                detected_text = detected_text.replace(" ", "")
                detected_text = detected_text.replace("x", "*")
                detected_text = detected_text.replace("X", "*")
                detected_text = detected_text.replace("÷", "/")

                print("Detected:", detected_text)

                # ============================================
                # SOLVE MATH
                # ============================================
                try:

                    result = eval(detected_text)

                    answer = str(result)

                    print("Answer:", answer)

                except Exception as e:

                    print("ERROR:", e)

                    answer = "INVALID"

    else:
        xp, yp = 0, 0

    # ============================================
    # MATCH CANVAS SIZE
    # ============================================
    canvas = cv2.resize(
        canvas,
        (img.shape[1], img.shape[0])
    )

    # ============================================
    # MERGE CANVAS + CAMERA
    # ============================================
    grayCanvas = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2GRAY
    )

    _, inv = cv2.threshold(
        grayCanvas,
        50,
        255,
        cv2.THRESH_BINARY_INV
    )

    inv = cv2.cvtColor(
        inv,
        cv2.COLOR_GRAY2BGR
    )

    img = cv2.bitwise_and(img, inv)

    img = cv2.bitwise_or(img, canvas)

    # ============================================
    # SMALL UI TEXT
    # ============================================
    cv2.putText(
        img,
        "PINCH = DRAW",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        img,
        "OPEN 2 FINGERS = MOVE",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2
    )

    cv2.putText(
        img,
        "THUMB UP = SOLVE",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        img,
        "ALL FINGERS = CLEAR",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # ============================================
    # ANSWER BOX
    # ============================================
    if answer != "":

        cv2.rectangle(
            img,
            (850, 500),
            (1250, 650),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            img,
            "ANSWER",
            (950, 560),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        cv2.putText(
            img,
            answer,
            (930, 630),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            5
        )

    # ============================================
    # SHOW WINDOW
    # ============================================
    cv2.imshow(
        "AI Gesture Math Solver",
        img
    )

    key = cv2.waitKey(1)

    # ESC TO EXIT
    if key == 27:
        break

# ============================================
# RELEASE
# ============================================
cap.release()
cv2.destroyAllWindows()