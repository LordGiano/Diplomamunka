from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Háttér kivonás képeken és videókon az OpenCV dokumentáció alapján. Színes megjelenítés.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='videos\\Rat2\\08.06.59-08.08.52[M][0@0][0].dav')
#'videos\\Rat2\\00.21.38-00.26.49[M][0@0][0].dav'
#22.20.51-22.27.45[M][0@0][0]
#08.06.59-08.08.52[M][0@0][0].dav
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).')  # KNN
args = parser.parse_args()

# Create Background Subtractor objects
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN(history=1500, dist2Threshold=3000, detectShadows=False)  # egész szép eredményt tud adni

# Capture the input video
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

# Read the first frame and initialize grid_mask with the same shape, set all pixels to black
ret, first_frame = capture.read()
if not ret:
    print('Unable to read the first frame')
    exit(0)

grid_mask = np.zeros_like(first_frame, dtype=np.uint8)  # Initializing grid_mask with all black pixels
prev_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)  # Convert first frame to grayscale

paused = False  # Variable to track if the video is paused

while True:
    if not paused:
        ret, frame = capture.read()
        if frame is None:
            break

        # Update the background model and get the foreground mask
        fgMask = backSub.apply(frame)

        # Convert the current frame to grayscale
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Compute the absolute difference between the current frame and the previous frame
        diff_frame = cv.absdiff(prev_frame, gray_frame)

        # Threshold the difference to highlight areas with significant change
        _, thresh_diff = cv.threshold(diff_frame, 30, 255, cv.THRESH_BINARY)

        # Update grid_mask where changes have been detected
        grid_mask[thresh_diff > 0] = [255, 255, 255]  # Set changed pixels to white in grid_mask

        # Update prev_frame to the current frame for the next iteration
        prev_frame = gray_frame

        # Extract the foreground in color
        fgColor = cv.bitwise_and(frame, frame, mask=fgMask)

        # Get the frame number and write it on the current frame
        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # Show the current frame, the color foreground mask, and the change detection mask
        cv.imshow('FG Mask', fgColor)
        cv.imshow('Fg mask', fgMask)
        cv.imshow('Grid Mask', grid_mask)  # Show the grid_mask

    # Wait for user input
    key = cv.waitKey(30)

    if key == ord('q') or key == 27:  # Exit if 'q' or ESC is pressed
        break
    elif key == ord(' '):  # Toggle pause/play on spacebar press
        paused = not paused

# Save the final grid_mask image to a file
cv.imwrite('grid_mask_output.png', grid_mask)

# Display the final grid_mask image
cv.imshow('Final Grid Mask', grid_mask)
cv.waitKey(0)

capture.release()
cv.destroyAllWindows()