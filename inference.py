import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 


model  = load_model("model.h5")
label = np.load("labels.npy")



holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 

# Load the model and labels
model  = load_model("model.h5")
label = np.load("labels.npy")

# Initialize MediaPipe solutions
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing = mp.solutions.drawing_utils

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Initialize the list to store landmarks
    lst = []

    # Read a frame from the camera
    _, frm = cap.read()

    # Flip the frame horizontally
    frm = cv2.flip(frm, 1)

    # Process the frame using MediaPipe
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Check if face landmarks are detected
    if res.face_landmarks:
        # Calculate the relative positions of face landmarks
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        # Check if left hand landmarks are detected
        if res.left_hand_landmarks:
            # Calculate the relative positions of left hand landmarks
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            # If not, append zeros
            for i in range(42):
                lst.append(0.0)

        # Check if right hand landmarks are detected
        if res.right_hand_landmarks:
            # Calculate the relative positions of right hand landmarks
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            # If not, append zeros
            for i in range(42):
                lst.append(0.0)

        # Convert the list to a numpy array and reshape it
        lst = np.array(lst).reshape(1,-1)

        # Make a prediction using the model
        pred = label[np.argmax(model.predict(lst))]

        # Print the prediction
        print(pred)

        # Draw the prediction on the frame
        cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

    # Draw the landmarks on the frame
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("window", frm)

    # Check for the 'esc' key to exit
    if cv2.waitKey(1) == 27:
        # Release the camera and close the window
        cv2.destroyAllWindows()
        cap.release()
        break


while True:
	lst = []

	_, frm = cap.read()

	frm = cv2.flip(frm, 1)

	res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))


	if res.face_landmarks:
		for i in res.face_landmarks.landmark:
			lst.append(i.x - res.face_landmarks.landmark[1].x)
			lst.append(i.y - res.face_landmarks.landmark[1].y)

		if res.left_hand_landmarks:
			for i in res.left_hand_landmarks.landmark:
				lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				lst.append(0.0)

		if res.right_hand_landmarks:
			for i in res.right_hand_landmarks.landmark:
				lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				lst.append(0.0)

		lst = np.array(lst).reshape(1,-1)

		pred = label[np.argmax(model.predict(lst))]

		print(pred)
		cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

		
	drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
	drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
	drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

	cv2.imshow("window", frm)

	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break

