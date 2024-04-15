import cv2

def detect_arrow(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arrow_detected = False
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        min_width = 20
        max_width = 100
        min_height = 20
        max_height = 100
        if min_width < w < max_width and min_height < h < max_height:
            arrow_detected = True
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if arrow_detected:
        print("Arrow detected")
    else:
        print("No arrow detected")

    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = detect_arrow(frame)

        cv2.imshow('Arrow Detection', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
