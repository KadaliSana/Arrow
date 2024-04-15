import cv2

def detect_arrow(frame, template_path):
    template = cv2.imread(template_path)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    arrow_detected = False
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter based on pixel dimensions (relative to 720p resolution)
        min_width = 100  # in pixels
        max_width = 700  # in pixels
        min_height = 100  # in pixels
        max_height = 700  # in pixels
        if min_width < w < max_width and min_height < h < max_height:
            arrow_detected = True
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    harris_response_image = cv2.cornerHarris(gray_image, blockSize=3, ksize=3, k=0.04)
    threshold_image = 0.01 * harris_response_image.max()
    harris_response_thresh_image = cv2.threshold(harris_response_image, threshold_image, 255, cv2.THRESH_BINARY)[1]
    num_corners_image = cv2.countNonZero(harris_response_thresh_image)

    harris_response_template = cv2.cornerHarris(gray_template, blockSize=3, ksize=3, k=0.04)
    threshold_template = 0.01 * harris_response_template.max()
    harris_response_thresh_template = cv2.threshold(harris_response_template, threshold_template, 255, cv2.THRESH_BINARY)[1]
    num_corners_template = cv2.countNonZero(harris_response_thresh_template)

    if arrow_detected and num_corners_image > num_corners_template:
        print("Arrow detected")

    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    template_path = "1.jpg"
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = detect_arrow(frame, template_path)

        cv2.imshow('Arrow Detection', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
