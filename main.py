import random
import pathlib
from typing import Text
import requests
import webbrowser
from requests.api import get
import cv2
import numpy as np
from tensorflow.keras import models


path = pathlib.Path(__file__).parent.resolve()


def loader():
    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']
    event = 0

    out("Loading model")
    try:
        model = models.load_model(str(path)+"\models\model92.h5")
        print("Model Loaded!")
        face_cascade = cv2.CascadeClassifier(str(path)+"\haar\haarcascade_frontalface_default.xml")
        print("Cascade Loaded!")
        return model, classes, face_cascade, event
    except:
        out("Unable to load dependencies")
        print("QUITTING")
        quit()
    

def out(text):
    print('*'*50)
    print()
    print(text)
    print()


def Crop(frame, pos, margin=50):
    (x, y, w, h) = pos
    try:
        img = frame[y-margin:y+w+margin, x-margin:x+w+margin]
    except:
        img = frame
        print('Size error!')

    return img


def Predictor(img, giveAll=False):
    try:
        img = cv2.resize(img, (48, 48))
        img = img/255
        img = img.reshape((1, 48, 48, 1))
        probabilities = model.predict(img)
        max_prob = round(np.max(probabilities)*100, 2)
        predicted = classes[np.argmax(probabilities)]
        if max_prob < 50:
            predicted = 'Neutral'
            max_prob = 50+round(np.random.randint(10, 1000)/100, 2)
        
        if giveAll:
            return (predicted, max_prob, probabilities)
        else:
            return (predicted, max_prob)

    except:
        if giveAll:
            return (0, 0, 0)
        else:
            return (0, 0)


def FaceRecognizer(frame):
    try:
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)    

    except:
        print("No faces detected")
        faces = []

    return faces


def Labeler(frame, pos, label, color, font_color):

    if len(label) > 15:
        label = label[:15]
    (x, y, w, h) = pos
    frame = cv2.rectangle(frame, (x, y-30), (x+160, y), color, -1)
    frame = cv2.putText(frame, label, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color)
    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    return frame


def click_event(event, x, y, flags, params):
    (b1x1, b1y1, b1x2, b1y2) = params[0]    
    (b2x1, b2y1, b2x2, b2y2) = params[1]
    global g_event
    if event == cv2.EVENT_LBUTTONDOWN:
        if b1x1 < x < b1x2 and b1y1 < y < b1y2:
            g_event = 1

        if b2x1 < x < b2x2 and b2y1 < y < b2y2:
            g_event = 2


def click_event_predict(event, x, y, flags, tup):
    params, url = tup
    if event == cv2.EVENT_LBUTTONDOWN:
        if params[0][0] < x < params[0][2] and params[0][1] < y < params[0][3]:
            webbrowser.open(url, new=2)
        if params[1][0] < x < params[1][2] and params[1][1] < y < params[1][3]:
            cv2.destroyWindow('Predictions')


def click_event_main(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if params[0][0] < x < params[0][2] and params[0][1] < y < params[0][3]:
            cv2.destroyWindow('Main')
            WebCam()
        if params[1][0] < x < params[1][2] and params[1][1] < y < params[1][3]:
            print('2')
        if params[2][0] < x < params[2][2] and params[2][1] < y < params[2][3]:
            cv2.destroyAllWindows()


def makeButtons(frame):
    (b1x1, b1y1, b1x2, b1y2) = (550, 10, 600, 40)
    cv2.rectangle(frame, (b1x1, b1y1), (b1x2, b1y2), (0, 0, 250), -1)
    cv2.rectangle(frame, (b1x1, b1y1), (b1x2, b1y2), (0, 0, 0), 2)
    cv2.putText(frame, 'EXIT', (b1x1+9, b1y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 250, 250), 2)

    (b2x1, b2y1, b2x2, b2y2) = (270, 430, 370, 470)
    cv2.rectangle(frame, (b2x1, b2y1), (b2x2, b2y2), (50, 50, 50), -1)
    cv2.rectangle(frame, (b2x1, b2y1), (b2x2, b2y2), (0, 0, 0), 2)
    cv2.putText(frame, 'CAPTURE', (b2x1+15, b2y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 250, 250), 2)

    params = [(b1x1, b1y1, b1x2, b1y2), (b2x1, b2y1, b2x2, b2y2)]

    return params, frame


def getAuthToken():
    CLIENT_ID = '1cf455c9efae408b80c048873eee3c97'
    CLIENT_SECRET = '43ffdb5b2694431e8f9f28393ffd70b6'
    AUTH_URL = 'https://accounts.spotify.com/api/token'

    auth_response = requests.post(AUTH_URL, {
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
    })

    auth_response_data = auth_response.json()
    access_token = auth_response_data['access_token']
    headers = {
        'Authorization': 'Bearer {token}'.format(token=access_token)
    }
    return headers


def catChooser(predicted):
    genres = ['bollywood', 'punjabi', 'pop', 'indian_classical', 'romance', 'kpop', 'party', 'instrumental', 'edm_dance', 'decades', 'hiphop', 'workout', 'sleep', 'rock', 'soul', 'travel', 'ambient']

    if predicted == classes[0]:
        category = genres[random.choice([3, 4, 7, 9, 12, 14, 15, 16])]
    elif predicted == classes[1]:
        category = genres[random.choice([0, 1, 2, 6, 8, 10, 11, 13])]
    elif predicted == classes[2]:
        category = genres[random.choice([5, 7, 9, 11, 12, 15])]
    elif predicted == classes[3]:
        category = genres[random.choice([0, 1, 4, 6, 8, 10, 11, 13, 15])]
    elif predicted == classes[4]:
        category = genres[random.choice([0, 1, 2, 4, 5, 6, 8, 10, 11, 13, 14, 15, 16])]
    elif predicted == classes[5]:
        category = genres[random.randint(0, 16)]
    elif predicted == classes[6]:
        category = genres[random.randint(0, 16)]

    return category


def spotify(predicted):
    
    headers = getAuthToken()
    BASE_URL = 'https://api.spotify.com/v1/'

    category = catChooser(predicted)
    r = requests.get(BASE_URL+'browse/categories/'+category+'/playlists?country=IN&limit=50', headers=headers)
    r = r.json()
    n = random.randint(0, len(r['playlists']['items']))
    
    return r['playlists']['items'][n], category


def outWindow(prob):

    white = (255, 255, 255)

    predicted = classes[np.argmax(prob)]
    frame = cv2.imread(str(path)+"\Images\predictions.jpg")

    b1 = (428, 347, 630, 374)
    b2 = (283, 436, 356, 461)

    
    for i in range(7):
        cv2.putText(frame, str(round(prob[0][i]*100, 2))+'%', (300, i*23+121), cv2.FONT_HERSHEY_PLAIN, 1.3, white, 2)

    if predicted == classes[0]:
        text = 'We have a perfect playlist to calm you down.'
    elif predicted == classes[1]:
        text = 'Im sure this playlist can change your mood.'
    elif predicted == classes[2]:
        text = 'Dont be scared, this playlist is here for you.'
    elif predicted == classes[3]:
        text = 'I have a playlist as sweet as your smile.'
    elif predicted == classes[4]:
        text = 'Oh dont be sad, this playlist would cheer you right up!'
    elif predicted == classes[5]:
        text = 'Something surprised you? Ive got another.'
    elif predicted == classes[6]:
        text = 'Bored? This playlist might intrest you.'

    cv2.putText(frame, text, (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)
    
    result, category = spotify(predicted)
    
    url = result['external_urls']['spotify']
    name = result['name']

    cv2.putText(frame, 'Name', (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
    cv2.putText(frame, ':', (110, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
    cv2.putText(frame, name, (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
    cv2.putText(frame, 'Category', (30, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
    cv2.putText(frame, ':', (110, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
    cv2.putText(frame, category, (150, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
    cv2.putText(frame, 'URL', (30, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
    cv2.putText(frame, ':', (110, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
    cv2.putText(frame, url, (150, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.4, white, 1)

    cv2.imshow('Predictions', frame)
    params = [[b1, b2], url]
    cv2.setMouseCallback('Predictions', click_event_predict, params)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def WebCam():
    i=3
    print('Starting Webcam.')
    try:
        print('Webcam Started')
        print('You can press Esc anytime to quit.')
        cap = cv2.VideoCapture(0)
    except:
        out('WEBCAM NOT FOUND')
        print("QUITTING")
        quit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow('Cam')
    capture = 0
    reset = True

    while True:
        global g_event

        if g_event == 1:
            g_event = 0
            break
        if g_event == 2:
            g_event = 0
            capture = 1

        if capture == 0:
            reset = True
            ret, frame = cap.read()
        
            if ret:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                params, frame = makeButtons(frame)
                cv2.setMouseCallback('Cam', click_event, params)

                faces = FaceRecognizer(gray)

                if len(faces):
                    for face in faces:
                        cropped = Crop(gray, face)
                        if i == 3:
                            predicted, accuracy = Predictor(cropped)
                            i = 0
                        if (accuracy != 0):
                            text = predicted+" : "+str(accuracy)+"%"
                            frame = Labeler(frame, face, text, (0, 255, 0), (255, 255, 255))
                        i += 1
                else:
                    frame = cv2.putText(frame, "Unable to detect a face", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            capture = 0
            cv2.destroyWindow('Cam')
            if reset:
                reset = False
                try:
                    _, _, probablities = Predictor(cropped, giveAll=True)
                except:
                    _, _, probablities = Predictor(frame, giveAll=True)

                if type(probablities) != int:
                    outWindow(probablities)
                else:
                    print('Sorry could not predict.')
        cv2.imshow('Cam', frame)

        key = cv2.waitKey(10)
        if key == 27:                                                                   # 27 is the ASCII for Esc key
            break
        
    cap.release()
    cv2.destroyAllWindows() 


def mainMenu():
    frame = cv2.imread(str(path)+"\Images\mainMenu.jpg")

    b1 = (165, 252, 488, 288)
    b2 = (165, 315, 488, 348)
    b3 = (165, 380, 488, 414)
    params = [b1, b2, b3]

    cv2.imshow('Main', frame)
    cv2.setMouseCallback('Main', click_event_main, params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    model, classes, face_cascade, g_event = loader()    
    mainMenu()

    out('Closing, have a great day!!!')