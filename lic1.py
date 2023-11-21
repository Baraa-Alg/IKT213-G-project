import cv2
import numpy as np
import imutils
import easyocr
import sys


def recognize_license_plate(cropped_image):
    try:
        # Initialize EasyOCR reader for English language (CPU mode)
        reader = easyocr.Reader(['en'], gpu=False)

        # Use OCR to read text from the cropped license plate image
        result = reader.readtext(cropped_image)

        # Check if OCR result is not empty
        if result:
            return result[0][1]
        else:
            return None
    except Exception as e:
        print(f"An error occurred during OCR: {str(e)}")
        return None


# Load the image
# img = cv2.imread('car1.png')
img = cv2.imread('car2.png')
# img = cv2.imread('car.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply bilateral filter to reduce noise and preserve edges
bfilter = cv2.bilateralFilter(gray, 11, 11, 17)

# Apply Canny edge detection to find edges in the image
edged = cv2.Canny(bfilter, 30, 500)


# Find contours in the edged image
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)

# Sort contours based on area in descending order and select the top 10
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None

# Iterate through contours and find the one with four points (license plate assumption)
for contour in contours:
    # Adjust epsilon for contour approximation (larger epsilon for a larger range)
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        location = approx
        break

# If a valid contour is found, proceed with OCR
if location is not None:
    # Draw contours on a black mask
    mask_img = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask_img, [location], 0, 255, -1)

    # Apply the mask to the original image
    cropped_image = gray[max(location[:, :, 1].min() - 10, 0):location[:, :, 1].max() + 10,
                         max(location[:, :, 0].min() - 10, 0):location[:, :, 0].max() + 10]

    # Recognize license plate
    license_plate_text = recognize_license_plate(cropped_image)

    if license_plate_text:
        # Redirect standard output to a file
        sys.stdout = open('output.txt', 'w')
        print("Detected text:", license_plate_text)
        sys.stdout.close()
        # Reset standard output to the console
        sys.stdout = sys.__stdout__
        print("Detected text:", license_plate_text)
else:
    print("No contours found.")
