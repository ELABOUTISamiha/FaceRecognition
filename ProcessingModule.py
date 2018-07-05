import cv2, os
import numpy as np
from PIL import Image
import collections
import time

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade, recognizer = None, None
last_names = None

CONTAINS_ONLY_FACES = True
NB_TRAINING_IMGS = 10
images_path = './images'


def etiquette(name, mat, color, origin, xsize, ysize):
    cv2.rectangle(mat, origin, (origin[0] + xsize, origin[1] + ysize), color, 2)
    cv2.putText(mat, name, (origin[0], origin[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,  color, 1)

def get_images_and_labels(path):
    # je recupere les CHEMINS des images de training
    training_imgs_paths = []
    for f in os.listdir(path):
        if int(f.split("_")[1].split(".")[0]) < NB_TRAINING_IMGS:
            training_imgs_paths.append(os.path.join(path, f))

    # list des IMAGES
    training_images = []
    #liste des LABELS (IDs)
    labels = []

    #if not CONTAINS_ONLY_FACES: # SI ON SUPPOSE QUE LES IMAGES CONTINNENT UNIQUEMENT DES VISAGES
    for image_path in training_imgs_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')

        # image ajoute'e
        training_images.append(image)
        subject_id = int(os.path.split(image_path)[1].split("_")[0])

        # l'ID ajoute'
        labels.append(subject_id)
    print labels
        #cv2.imshow("TRAINING - %s" % image_path, image)   #subject_id, image)
        #cv2.waitKey(10)
    # else:
    #     for image_path in training_imgs_paths:
    #         #image_pil = Image.open(image_path).convert('L')
    #         #image = np.array(image_pil, 'uint8')
    #         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #         faces = faceCascade.detectMultiScale(image)
    #         # If face is detected, append the face to images and the label to labels
    #         for (x, y, w, h) in faces:
    #             training_images.append(image[y: y + h, x: x + w])
    #             subject_id = int(os.path.split(image_path)[1].split("_")[0])
    #             labels.append(subject_id)
    #             #cv2.imshow("TRAINING - %s" % image_path, image[y: y + h, x: x + w])
    #             cv2.imshow("TRAINING - %d" % subject_id, image[y: y + h, x: x + w])
    #             cv2.waitKey(2)

    return training_images, labels


def train_recognizer():
    global cascadePath, faceCascade, recognizer
    faceCascade = cv2.CascadeClassifier(cascadePath)

    # For face recognition we will the the LBPH Face Recognizer
    recognizer = cv2.createLBPHFaceRecognizer()

    # recognizer = cv2.createEigenFaceRecognizer()    #EigenFace needs images of same size
    images, labels = get_images_and_labels(images_path)
    cv2.destroyAllWindows()

    # Perform the tranining
    recognizer.train(images, np.array(labels))


def run_recognizer():
    global recognizer, last_names

    # Last_names est un dictionnaire
    # key value
    # {1 , "Samiha"}
    # {2, "Nour"}
    # names[1] = "el samiha"
    last_names = {}
    last_names[1] = "EL ABOUTI"
    last_names[2] = "AIT SAID"
    last_names[3] = "Prof. BOUYAHYAOUI"

    cap = cv2.VideoCapture(0)

    count = 0
    results = {}
    #presents_list = set()
    while 1:
        _, predict_image = cap.read()

        # Our operations on the frame come here
        # c'est l'image que je doit pre'dire
        predict_image = cv2.cvtColor(predict_image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(predict_image)

        if len(faces) > 0:
            max_area = 0
            x = y = w = h = 0
            for (_x, _y, _w, _h) in faces:
                a = _w * _h
                if a > max_area:
                    max_area = a
                    x, y, w, h = _x, _y, _w, _h
            cv2.rectangle(img=predict_image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            # on pre'dit uniquement la face la plus grande
            nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])

            # get -->> gets 0 if the predicted id doesn't exist initially then ++
            results[nbr_predicted] = results.get(nbr_predicted, 0) + 1
            count += 1
            #print "{} is Recognized with confidence {}".format(last_names[nbr_predicted]), conf)
            print "{} est reconnue avec un coefficient = {}".format(last_names[nbr_predicted], conf)

	    if count == 11 :
                c = collections.Counter(results)
                res = c.most_common(1)[0][0] #celui qui a e'te' de'te'dte' le plus
                print "---------------->>>>>   {}   <<<<<<<----------------".format(last_names[res])
                etiquette(name=last_names[res], mat=predict_image, color=(255,0,0), origin=(x, y), xsize=w, ysize=h)
                count = 0
                results = {}

            cv2.imshow("Recognizing Face", predict_image)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyAllWindows()
