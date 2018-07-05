import cv2

class ImageDbManager():
    def __init__(self, id, cascadePath="haarcascade_frontalface_default.xml", NB_TRAINING_IMGS = 10, WIDTH = 180, HEIGHT=240):
        self.id = id
        self.cpt = 0
        self.WIN_TITLE = "Adding " + str(id)
        self.NB_TRAINING_IMGS = NB_TRAINING_IMGS
        self.IMG_FOLDER = "images"
        self.faceCascade = cv2.CascadeClassifier(cascadePath)

        self.FACE_WIDTH = WIDTH
        self.FACE_HEIGHT = HEIGHT

        self.width = None
        self.heigth = None

        cv2.namedWindow(self.WIN_TITLE)

    def insert_cpt(self, imge):
        cv2.rectangle(imge, (30,30), (185, 60), (255, 0,0) , 3)
        cv2.putText(imge, "Taken photos " + str(self.cpt) , (45,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))

    def insert_rec(self, img):
        cv2.rectangle(img=img, pt1=((self.width - self.FACE_WIDTH)/2 , (self.heigth - self.FACE_HEIGHT)/2), pt2=((self.width + self.FACE_WIDTH)/2 , (self.heigth + self.FACE_HEIGHT)/2),
                      color=(255, 255, 255), thickness=4)

    def save_img(self, img):
        cropped = img[(self.heigth/2 - self.FACE_HEIGHT/2) : (self.heigth/2 + self.FACE_HEIGHT/2), (self.width/2 - self.FACE_WIDTH/2) : (self.width/2 + self.FACE_WIDTH/2)]
        cv2.imshow("apercu" , cropped)
        cv2.imwrite('{}/{}_{}.jpg'.format(self.IMG_FOLDER, str(self.id), self.cpt), cropped)

    def run(self):
        cap = cv2.VideoCapture(0)
        _, img = cap.read()
        self.width = img.shape[1]
        print self.width
        self.heigth = img.shape[0]
        print self.heigth
        while 1:
            _, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.insert_cpt(img)
            self.insert_rec(img)
            cv2.imshow(self.WIN_TITLE , img)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('s') :
                _, img = cap.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.faceCascade.detectMultiScale(img)
                if len(faces) > 0:
                    self.save_img(img)
                    self.cpt += 1
                    print(self.cpt)

            if k == ord('q') and (self.cpt >= self.NB_TRAINING_IMGS) :
                break
        cv2.destroyAllWindows()
