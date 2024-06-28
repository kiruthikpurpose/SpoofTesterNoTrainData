import cv2
import dlib
import numpy as np

# Load face detector from dlib
detector = dlib.get_frontal_face_detector()

# Load the pre-trained face landmark predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor = dlib.shape_predictor(predictor_path)

def get_face_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    face = faces[0]
    landmarks = predictor(gray, face)
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

    return landmarks

def is_real_face(landmarks):
    # Simple check: You can implement more sophisticated checks based on your needs.
    # For example, you can use machine learning models for classification.

    # Calculate the distance between certain facial landmarks (e.g., eyes)
    eye_distance_threshold = 20
    left_eye_distance = np.linalg.norm(landmarks[42] - landmarks[45])
    
    return left_eye_distance > eye_distance_threshold

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from webcam")
            break

        landmarks = get_face_landmarks(frame)

        if landmarks is not None:
            if is_real_face(landmarks):
                cv2.putText(frame, "Real Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Fake Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Anti-Spoofing System", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()