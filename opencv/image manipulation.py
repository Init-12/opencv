import cv2


def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def kadrir():
    image = cv2.imread("stol.jpg")
    cropped = image[10:500, 500:2000]
    viewImage(cropped, "После кадирования")

def resized():
    image = cv2.imread("stol.jpg")
    scale_percent = 20  # Процент от изначального размера
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    viewImage(resized, "После изменения размера на 20%")

def turn():
    image = cv2.imread("stol.jpg")
    (h, w, d) = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    viewImage(rotated, "После поворота на 180 градусов")

def grad():
    image = cv2.imread("stol.jpg")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(image, 127, 255, 0)
    viewImage(gray_image, "В градациях серого")
    viewImage(threshold_image, "Чёрно-белый")

def rz():
    image = cv2.imread("stol.jpg")
    blurred = cv2.GaussianBlur(image, (51, 51), 0)
    viewImage(blurred, "Размытый")

def pr():
    image = cv2.imread("stol.jpg")
    output = image.copy()
    cv2.rectangle(output, (380, 200), (280, 100), (0, 255, 255), 10)
    viewImage(output, "Обводим прямоугольником")

def drlineas():
    image = cv2.imread("stol.jpg")
    output = image.copy()
    cv2.line(output, (360, 100), (400, 200), (0, 0, 255), 5)
    viewImage(output, "Линия")

def text():
    image = cv2.imread("stol.jpg")
    output = image.copy()
    cv2.putText(output, "Table", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (30, 105, 210), 10)
    viewImage(output, "Изображение с текстом")

def facerecognition():
    image_path = "face.jpg"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10, 10)
    )
    faces_detected = "Лиц обнаружено: " + format(len(faces))
    print(faces_detected)
    # Рисуем квадраты вокруг лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
    viewImage(image, faces_detected)

def face():
    image_path="face1.jpg"
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image=cv2.imread(image_path)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10,10)
    )
    faces_detected="faces detected : "+format(len(faces))
    print(faces_detected)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),30)
    viewImage(image,faces_detected)
# pr()
# kadrir()
# turn()
# grad()
# rz()
# drlineas()
# facerecognition()\
face()