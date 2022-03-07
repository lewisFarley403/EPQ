import cv2
import os
face_cascade = cv2.CascadeClassifier(
    r'haarcascade_frontalface_default.xml')
initDir = 'masked'


def show_webcam(mirror=True):
    i = 0
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
            # print('writing')
            cv2.imwrite(f'{initDir}/imgs/{str(i)}.jpg', img)
        cv2.imshow('my webcam', img)
        i += 1
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def webcam():
    show_webcam(mirror=True)


def faceFrame():
    mirror = True
    i = 0
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()

        img = cv2.flip(img, 1)

        show = img
        wh = 120
        cv2.line(show, (show.shape[1]//2+wh, show.shape[0]//2+wh),
                 (show.shape[1]//2+wh, show.shape[0]//2-wh), (0, 0, 255))

        cv2.line(show, (show.shape[1]//2+wh, show.shape[0]//2-wh),
                 (show.shape[1]//2-wh, show.shape[0]//2-wh), (0, 0, 255))

        cv2.line(show, (show.shape[1]//2+wh, show.shape[0]//2+wh),
                 (show.shape[1]//2-wh, show.shape[0]//2+wh), (0, 0, 255))
        # print('writing')
        # cv2.imwrite(f'{initDir}/imgs/{str(i)}.jpg', img)
        cv2.imshow('my webcam', show)
        i += 1
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def process_img(image):
    original_image = image
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    return processed_img


def faceSave():
    width = 120
    height = width
    i = 0
    for file in os.listdir(f'{initDir}/imgs'):
        if file.endswith('.jpg'):
            img = cv2.imread(f'{initDir}/imgs/{file}')
            #new_screen = process_img(screen)
            screen = img
            #print('img loaded')
            faces = face_cascade.detectMultiScale(screen, 1.35, 5)
            #print('face detected')
            if len(faces) == 0:
                print('no faces detected')
            for (x, y, w, h) in faces:

                #print('in face')
                cv2.rectangle(screen, (x, y), (x+w, y+h), (255, 255, 0), 2)
                roi_gray = screen[y:y+h, x:x+w]
                roi_color = screen[y:y+h, x:x+w]
                roi_color = cv2.resize(
                    roi_color, (width, height), interpolation=cv2.INTER_AREA)
                cv2.imwrite(f'{initDir}/imgs/roi/{str(i)}.jpg', roi_color)
            # print('done')
            #cv2.imshow('img', roi_color)

            if cv2.waitKey(1) == 27:
                break  # esc to quit
        i += 1
    cv2.destroyAllWindows()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    # faceSave()
    faceFrame()
