import os
import cv2
import numpy as np
from django.views.decorators.csrf import csrf_protect
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import date
from django.shortcuts import render,HttpResponse,redirect
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from django.contrib.auth.models import User
import os
import csv
from django.http import HttpResponse
from django.shortcuts import render
from datetime import date, datetime




import csv
from django.http import HttpResponse

def home(request):
    names, rolls, times, l = extract_attendance()
    csv_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Attendance'))
    print(f"Absolute path to 'Attendance' folder: {csv_folder}")

    # Get a list of CSV files in the 'Attendance' folder
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    return render(request, 'home.html', {'names': names, 'rolls': rolls, 'times': times, 'l': l, 'totalreg': totalreg(), 'datetoday2': datetoday2, 'mess': MESSAGE,'csv_files': csv_files})

def add(request):
    if request.method == 'POST':
        newusername = request.POST.get('newusername')
        id=request.POST.get('id')
        userimagefolder = 'static/faces/' + newusername
        user = User(username=newusername, id=id)
        user.save()
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        cap = cv2.VideoCapture(0)
        i, j = 0, 0
        while 1:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 10 == 0:
                    name = newusername + '_' + str(i) + '.jpg'
                    cv2.imwrite(os.path.join(userimagefolder, name), frame[y:y+h, x:x+w])
                    i += 1
                j += 1
            if j == 500:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        print('Training Model')
        train_model(request)
        if totalreg() > 0:
            names, rolls, times, l = extract_attendance()
            MESSAGE = 'User added successfully'
            print("message changed")
            return render(request, 'home.html', {'names': names, 'rolls': rolls, 'times': times, 'l': l, 'totalreg': totalreg(), 'datetoday2': datetoday2, 'mess': MESSAGE})
        else:
            return redirect('home.html', {'names': names, 'rolls': rolls, 'times': times, 'l': l, 'totalreg': totalreg(), 'datetoday2': datetoday2, 'mess': MESSAGE})


def download_csv(request, file_path):
    # Construct the full file path
    full_path = os.path.join('Attendance', file_path)

    # Check if the file exists
    if os.path.exists(full_path):
        with open(full_path, 'rb') as csv_file:
            response = HttpResponse(csv_file.read(), content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="{file_path}"'
            return response
    else:
        return HttpResponse("File not found", status=404)
    
def start(request):

    ATTENDENCE_MARKED = False
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        names, rolls, times, l = extract_attendance()
        MESSAGE = 'This face is not registered with us , kindly register yourself first'
        print("face not in database, need to register")
        return render(request, 'home.html', {'names': names,'rolls': rolls,'times': times,'l': l,'totalreg': totalreg(),'datetoday2': datetoday2,'mess': MESSAGE})
        # return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            cv2.putText(frame, f'{identified_person}', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
            if cv2.waitKey(1) == ord('a'):
                add_attendance(identified_person)
                current_time_ = datetime.now().strftime("%H:%M:%S")
                print(f"attendence marked for {identified_person}, at {current_time_} ")
                ATTENDENCE_MARKED = True
                break
        if ATTENDENCE_MARKED:
            # time.sleep(3)
            break
        
        # Display the resulting frame
        cv2.imshow('Attendance Check, press "q" to exit', frame)
        cv2.putText(frame,'hello',(30,30),cv2.FONT_HERSHEY_COMPLEX,2,(255, 255, 255))
        
    # Wait for the user to press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    MESSAGE = 'Attendence taken successfully'
    print("attendence registered")
    return render(request, 'home.html', {'names': names,'rolls': rolls,'times': times,'l': l,'totalreg': totalreg(),'datetoday2': datetoday2,'mess': MESSAGE})

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times,l

def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces(img):
    if img != []:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def get_images_and_labels(request):
    face_images = []
    face_labels = []
    userlist = os.listdir('static/faces')

    for i, username in enumerate(userlist):
        for img in os.listdir(f'static/faces/{username}'):
            face_images.append(cv2.imread(f'static/faces/{username}/{img}', cv2.IMREAD_GRAYSCALE))
            face_labels.append(i)
    return face_images, face_labels


def train_model(request):
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')



def add_attendance(name):
    username = name.split('_')[0]
    id = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if str(id) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{id},{current_time}')
    else:
        print("This user has already marked attendance for the day, but still marking it.")





#VARIABLES
MESSAGE = "WELCOME  " \
        "Instruction: to register your attendence kindly click on 'a' on keyboard"



#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')

#### get a number of total registered users

#### extract the face from an image








