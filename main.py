import cv2

def detect_arrow(frame, template_path):
    template = cv2.imread(template_path)

    # Convert frame and template to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply binary thresholding
    _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours and detect arrows
    arrow_detected = False
    for contour in contours:
        # Calculate bounding box dimensions
        x, y, w, h = cv2.boundingRect(contour)
        # Filter based on pixel dimensions (relative to 720p resolution)
        min_width = 100  # in pixels
        max_width = 700  # in pixels
        min_height = 100  # in pixels
        max_height = 700  # in pixels
        if min_width < w < max_width and min_height < h < max_height:
            arrow_detected = True
            # Draw bounding box around contour
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Compute Harris corners for the frame
    harris_response_image = cv2.cornerHarris(gray_image, blockSize=3, ksize=3, k=0.04)
    threshold_image = 0.01 * harris_response_image.max()
    harris_response_thresh_image = cv2.threshold(harris_response_image, threshold_image, 255, cv2.THRESH_BINARY)[1]
    num_corners_image = cv2.countNonZero(harris_response_thresh_image)

    # Compute Harris corners for the template
    harris_response_template = cv2.cornerHarris(gray_template, blockSize=3, ksize=3, k=0.04)
    threshold_template = 0.01 * harris_response_template.max()
    harris_response_thresh_template = cv2.threshold(harris_response_template, threshold_template, 255, cv2.THRESH_BINARY)[1]
    num_corners_template = cv2.countNonZero(harris_response_thresh_template)

    # If arrow is detected based on contours and Harris corners, print message
    if arrow_detected and num_corners_image > num_corners_template:
        print("Arrow detected")

    return frame

if __name__ == "__main__":
    # Initialize video capture object
    cap = cv2.VideoCapture(0)

    # Set webcam resolution to 720p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    template_path = "1.jpg"
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Detect arrow
        result_frame = detect_arrow(frame, template_path)

        # Display result
        cv2.imshow('Arrow Detection', result_frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()
