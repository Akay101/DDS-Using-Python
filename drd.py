import cv2
from scipy.spatial import distance
import sounddevice as sd
from scipy.io import wavfile

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


cv2.namedWindow("Drowsiness Detection")

cv2.setWindowProperty("Drowsiness Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_GUI_NORMAL)


# Constants for eye aspect ratio thresholds
EAR_THRESHOLD = 1.3  # Adjust this value based on sensitivity

# Load the face cascade XML file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# Load the alarm sound
sample_rate, alarm_sound = wavfile.read('beep.wav')

# Start the video capture
video_capture = cv2.VideoCapture(0)

# Define a flag to track if the alarm is currently playing
alarm_playing = False



try:
    # Loop over frames from the video stream
    while True:
        # Read a frame from the video stream
        ret, frame = video_capture.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop over the detected faces
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Extract the region of interest (ROI) within the face rectangle
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detect eyes within the face ROI
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Loop over the detected eyes
            for (ex, ey, ew, eh) in eyes:
                # Draw rectangles around the eyes
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

                # Calculate the eye aspect ratio (EAR)
                eye = roi_gray[ey:ey+eh, ex:ex+ew]
                ear = eye_aspect_ratio(eye)

                # Display the eye aspect ratio on the frame
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Check if the eye aspect ratio is below the threshold
                if ear < EAR_THRESHOLD:
                    # Drowsiness detected
                    cv2.putText(frame, "Drowsy", (x, y-40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # If the alarm is not already playing, start playing the alarm sound
                    if not alarm_playing:
                        sd.play(alarm_sound, sample_rate)
                        alarm_playing = True
                else:
                    # If the eyes are not drowsy, stop playing the alarm
                    if alarm_playing:
                        sd.stop()
                        alarm_playing = False

        # Display the resulting frame
        cv2.imshow("Drowsiness Detection", frame)

        # Check for key press and break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except ValueError as e: 
        raise RuntimeError('error encountered') from e

finally:
    # Release the video capture object and close all OpenCV windows
    video_capture.release()
    cv2.destroyWindow("Drowsiness Detection")
    cv2.destroyAllWindows()
