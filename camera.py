import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Prepare data generator for standardizing frames before sending them into the model.
data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

# Loading the model.
MODEL_NAME = 'models/asl_alphabet_{}.h5'.format(9575)
model = load_model(MODEL_NAME)

# Setting up the input image size and frame crop size.
IMAGE_SIZE = 200
CROP_SIZE = 400

# Creating list of available classes stored in classes.txt.
classes_file = open("classes.txt")
classes_string = classes_file.readline()
classes = classes_string.split()
classes.sort()  # The predict function sends out output in sorted order.

# To detect hands in the video and to return the coordinates to crop the hand.


class handDetector():
    def __init__(self, mode = False, maxHands = 2, modelComplexity=1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    # To obtain and draw the hand landmarks
    def findHands(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    # Returns coordinates to crop the hand
    def findPosition(self, img, handNo = 0, draw = True):
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            cxList = []
            cyList = []
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cxList.append(cx)
                cyList.append(cy)
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
            return min(cxList), max(cxList), min(cyList),max(cyList)
        else:
            return w/2,w/2,h/2,h/2
#cap = cv2.VideoCapture(0)
detector = handDetector()
xMinList = []
xMaxList = []
yMinList = []
yMaxList = []        
class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,img=self.video.read()
        if ret:
            f_img = cv2.flip(img,1)
            img = cv2.flip(img,1)
            h_img = detector.findHands(f_img)
            xMin, xMax, yMin, yMax = detector.findPosition(h_img)
            xMinList.append(xMin)
            xMaxList.append(xMax)
            yMinList.append(yMin)
            yMaxList.append(yMax)
            # Cropping each frame as per the hand coordinates
            crop_img = img[int(max(yMin-30,0)):int(min(yMax+30,1920)), int(max(xMin-40,0)):int(min(xMax+30,1080)),:]
            #cropped_resized_img = cv2.resize(crop_img, (200, 200))
            #preprocessing for prediction
            #cropped_image = img[0:CROP_SIZE, 0:CROP_SIZE]
            crop_img = cv2.flip(crop_img,1)
            resized_frame = cv2.resize(crop_img, (IMAGE_SIZE, IMAGE_SIZE))
            reshaped_frame = (np.array(resized_frame)).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
            frame_for_model = data_generator.standardize(np.float64(reshaped_frame))

            img = cv2.flip(img,1)
            # Predicting the frame.
            prediction = np.array(model.predict(frame_for_model))
            predicted_class = classes[prediction.argmax()]      # Selecting the max confidence index.

            # Preparing output based on the model's confidence.
            prediction_probability = prediction[0, prediction.argmax()]
            if prediction_probability > 0.5:
            # High confidence.
                cv2.putText(img, '{} - {:.2f}%'.format(predicted_class, prediction_probability * 100), 
                                    (10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)
            elif prediction_probability > 0.2 and prediction_probability <= 0.5:
            # Low confidence.
                cv2.putText(img, 'Maybe {}... - {:.2f}%'.format(predicted_class, prediction_probability * 100), 
                                    (10, 450), 1, 2, (0, 255, 255), 2, cv2.LINE_AA)
            else:
            # No confidence.
                cv2.putText(img, classes[-2], (10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)
        
        ret,jpg=cv2.imencode('.jpg',img)
        return jpg.tobytes()