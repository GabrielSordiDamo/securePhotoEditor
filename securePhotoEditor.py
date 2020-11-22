from Stack import Stack
import mahotas
import numpy as np
import cv2
import os
import re
import glob
import time
import shutil
import face_recognition
import logging


# colorUtils*********************************************************
def loadPhotosPath(finalPoint):
    currentDir = os.getcwd()
    path = os.path.join(currentDir, finalPoint)
    photos = []
    photos = [p for p in glob.glob(path + '*.jpg')]
    return photos


def drawRectangleOnImg(img, initialPos, finalPos, color=(0, 0, 0)):
    cv2.rectangle(img, initialPos, finalPos, color, 1)


def getColorCountours(img, mask):

    res = cv2.bitwise_and(img, img, mask=mask)

    _, thrshed = cv2.threshold(cv2.cvtColor(
        res, cv2.COLOR_BGR2GRAY), 3, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thrshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours

# pathUtils***********************************************


def parsetPathToFinalPoint(path, preFinalExtension):
    exp = '(.{1,}' + f'{preFinalExtension})'
    sub = re.sub(exp, '', path, count=1, flags=re.I)
    return sub


def parsePathToName(path, preFinalExtension='known\\\\'):
    name = parsetPathToFinalPoint(path, preFinalExtension)
    name = re.sub('(.jpg)', '', name, count=1, flags=re.I)
    return name

# imgUtils**************************************************


def getGrayImg(img):
    imgCopy = img.copy()
    bgrImg = None
    try:
        bgrImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return imgCopy
    return bgrImg


def getBlurImg(img):
    imgCopy = img.copy()
    blurImg = None
    try:
        blurImg = cv2.blur(img, (20, 20))
    except Exception:
        return imgCopy
    return blurImg


def getBinImg(img):
    grayImg = getGrayImg(img)
    blurImg = getBlurImg(grayImg)
    imgCopy = img.copy()
    binImg = None
    try:
        T = mahotas.thresholding.otsu(blurImg)
        binImg = blurImg.copy()
        binImg[binImg >= T] = 255
        binImg[binImg < 255] = 0
        binImg = cv2.bitwise_not(binImg)
    except Exception:
        return imgCopy
    return binImg


def getCannyImg(img):
    binImg = getBinImg(img)
    imgCopy = img.copy()
    cannyImg = None
    try:
        cannyImg = cv2.Canny(binImg, 70, 150)
    except Exception:
        return imgCopy
    return cannyImg


def getMorphoOpeningImg(img):
    imgCopy = img.copy()
    openingImg = None
    try:
        kernel = np.ones((5, 5), np.uint8)
        openingImg = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    except Exception:
        return imgCopy
    return openingImg


def getMorphoCloseImg(img):
    imgCopy = img.copy()
    openingImg = None
    try:
        kernel = np.ones((5, 5), np.uint8)
        openingImg = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    except Exception:
        return imgCopy
    return openingImg


def getCropImg(img):
    os.system('cls')
    x, y, w, h = 0, 0, 0, 0
    while x == 0 or y == 0 or w == 0 or h == 0:
        print('ensira os valores para:')
        print('x, y, w, h')
        print('um depois do outro separados por virgula')
        x, y, w, h = input().split(',')
        x, y, w, h = int(x), int(y), int(w), int(h)
    img = img[y:y+h, x:x+w]
    return img


def flipOverXAxis(img):
    return cv2.flip(img, 1)

# mainPartsOfTheApp******************************************************


def getFaceImg():
    camera = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while True:
        _, frame = camera.read()
        frame = flipOverXAxis(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facesOnTheFrame = faceCascade.detectMultiScale(gray, 1.80, 5)
        x, y, w, h = 0, 0, 0, 0
        erasableImg = frame.copy()
        for (x, y, w, h) in facesOnTheFrame:
            drawRectangleOnImg(erasableImg, (x-50, y-50),
                               (x+w+50, y+h+50), color=(0, 0, 255))
        cv2.imshow("Camera", erasableImg)
        k = cv2.waitKey(30)
        if k == 27:
            exit(0)
        if k == ord('p'):
            if len(facesOnTheFrame) > 1 or len(facesOnTheFrame) == 0:
                print('imagem invalida mais de uma pessoa na imagem')
                print('ou nao ha nem uma pessoa na imagem ')
            else:
                cv2.destroyAllWindows()
                camera.release()
                return frame


def registerNewUser():

    print('um quadrado vermelho aparecera assim')
    print('que a imagem for valida')
    print('cliquem em: ')
    print('esc para sair do programa')
    print('p para salvar a imagem')
    frame = getFaceImg()
    userName = None
    while not userName:
        userName = input('digite seu nome: ')
    saveOn = f'{userName}.jpg'
    cv2.imwrite(saveOn, frame)
    shutil.move(f'./{saveOn}', f'./known/{saveOn}')
    os.system('cls')
    logging.info(f' new user registered {userName}')
    print('novo usuario cadastrado com sucesso')
    print('mas e necessario reiniciar o programa')
    exit(0)


def loadKnownUsers():
    currentDir = os.getcwd()
    path = os.path.join(currentDir, 'known\\')
    knownUsersImgsPath = []
    knownUsersImgsPath = [p for p in glob.glob(path + '*.jpg')]
    faceEncodings = []
    if not knownUsersImgsPath:
        print('nem um usuario encontrado')
        print('registrando novo usuario')
        time.sleep(4)
        os.system('cls')
        registerNewUser()
    else:
        for i in range(len(knownUsersImgsPath)):
            userImg = face_recognition.load_image_file(knownUsersImgsPath[i])
            encoding = None
            try:
                encoding = face_recognition.face_encodings(userImg)[0]
            except IndexError:
                # log
                print('nao foi possivel detectar nem uma face na imagem: ')
                print(knownUsersImgsPath[i])
                os.remove(knownUsersImgsPath[i])
                print('a imagem foi removida')
                print('encerrando programa')
                exit(0)
            faceEncodings.append(encoding)
        return faceEncodings, knownUsersImgsPath


def authenticateUser(usersEncoding, knownUsersImgsPath):
    userName = None
    while True:
        print('vamos tirar uma foto sua para identificar')
        print('voce entre nossos usuarios')
        print('um quadrado vermelho aparecera assim')
        print('que a imagem for valida')
        print('aperte: ')
        print('esc para sair do programa')
        print('p para prosseguir com a validacao')
        frame = getFaceImg()
        cv2.imwrite('temp.jpg', frame)
        frame = face_recognition.load_image_file('temp.jpg')
        os.remove('temp.jpg')
        frameEncoding = None
        try:
            frameEncoding = face_recognition.face_encodings(frame)[0]
        except IndexError:
            print('ocorreu um erro, vamos tentar novamente! ')
            time.sleep(2)
            os.system('cls')
            continue
        print('procurando match')
        print('(isso pode demorar um pouco)')
        for i in range(len(usersEncoding)):
            matchFaces = None
            matchFaces = face_recognition.compare_faces(
                [usersEncoding[i]], frameEncoding)
            if matchFaces[0] == True:
                userName = parsePathToName(knownUsersImgsPath[i])
                print(f'seja-bemvindo(a) {userName}')
                time.sleep(3)
                logging.info(f' loging {userName}')
                os.system('cls')
                return userName

        option = None
        while True:
            print('nao foi possivel identificalo entre os usuarios')
            print('digite:')
            print('x para sair do programa')
            print('n para cadastrar novo usuario')
            print('p bater outra foto: ')
            option = input()
            if option == 'x':
                exit(0)
            if option == 'n':
                os.system('cls')
                registerNewUser()
            if option == 'p':
                os.system('cls')
                break

# appUtils**************************************************************************


def getNextPhoto(path):
    imgName = parsetPathToFinalPoint(path, 'rawPhotos\\\\')
    shutil.copy(path, f'{imgName}')
    img = cv2.imread(imgName)
    os.remove(imgName)
    return img, imgName


def saveImgs(stackOfImgsToSave):
    while len(stackOfImgsToSave):
        img, imgName = stackOfImgsToSave.pop()
        cv2.imwrite(f'{imgName}', img)
        shutil.move(f'./{imgName}', f'./edithedPhotos/{imgName}')

# app********************************************************88


def app(userName):
    option = None
    while option != 'c' and option != 'x':
        os.system('cls')
        print('antes de editar as fotos vamos carregalas')
        print('c - para carregar as fotos da pasta photos')
        print('***********************************')
        print(' as fotos dever ter extensao .jpg')
        print('***********************************')
        print('x - para sair do programa')
        option = input()

    if option == 'x':
        print(f'esperamos te ver novamente {userName}')
        exit(0)
    photosPath = None
    if option == 'c':
        photosPath = loadPhotosPath('rawPhotos\\')

    if not photosPath:
        print('nao ha fotos em /photos')
        print('coloque algumas fotos la  e reinicie o programa')
        exit(0)

    modificateImg = {
        'c': getMorphoCloseImg,
        'o': getMorphoOpeningImg,
        'g': getGrayImg,
        'r': getBlurImg,
        'b': getBinImg,
        'k': getCannyImg
    }
    i = 0
    imgBeingEdited, imgBeingEditedName = getNextPhoto(photosPath[i])
    editingHistory = Stack(100)
    alreadyEditedImgs = Stack(100)
    print(f'{userName}, voce ja pode comecar a editar suas fotos')

    camera = cv2.VideoCapture(0)
    lowerPink = np.array([140, 130, 130])
    upperPink = np.array([180, 255, 255])
    _, frame = camera.read()
    cv2.imshow(f'{imgBeingEditedName}', imgBeingEdited)
    while True:
        os.system('cls')
        _, frame = camera.read()
        frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pinkMask = cv2.inRange(frameHsv, lowerPink, upperPink)
        pinkContours = getColorCountours(frame, pinkMask)
        print('mostre um objeto rosa se quiser sair do programa')
        print(f'editando foto {imgBeingEditedName}')
        print('aperte:')
        print('n - para carregar a poxima foto')
        print('c - para aplicar morpho-close')
        print('o - para aplicar morpho open')
        print('g - para mudar a cor da imagem para cinza')
        print('r - para embacar a imagem')
        print('k - para aplicar kenny')
        print('u - desfazer ultima modificacao')
        print('esc - sair do programa manualmente')

        for cnt in pinkContours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if area > 2500:
                if len(editingHistory) >= 1:
                    alreadyEditedImgs.push(
                        (imgBeingEdited, imgBeingEditedName))
                if len(alreadyEditedImgs) >= 1:
                    saveImgs(alreadyEditedImgs)
                    print(f'{userName} imagens savas em ./edithedPhotos')
                    logging.info(
                        f' {len(alreadyEditedImgs)} photos edited by {userName}')
                print('ate mais')
                logging.info(f' logoff {userName}')
                exit(0)

        k = cv2.waitKey(100)
        if k == 27:
            if len(editingHistory) >= 1:
                alreadyEditedImgs.push((imgBeingEdited, imgBeingEditedName))
            if len(alreadyEditedImgs) >= 1:
                saveImgs(alreadyEditedImgs)
                print(f'{userName} imagens savas em ./edithedPhotos')
                logging.info(
                    f' {len(alreadyEditedImgs)} photos edited by {userName}')
            print('ate mais')
            logging.info(f' logoff {userName}')
            exit(0)
        elif k == ord('n'):
            try:
                copyI = i + 1
                path = photosPath[copyI]
                copyImg = imgBeingEdited.copy()
                copyImgName = imgBeingEditedName
                imgBeingEdited, imgBeingEditedName = getNextPhoto(path)
                alreadyEditedImgs.push((copyImg, copyImgName))
                cv2.destroyAllWindows()
                cv2.imshow(f'{imgBeingEditedName}', imgBeingEdited)
                editingHistory = Stack(100)
            except IndexError:
                print('nao ha mais imagens para editar')
                time.sleep(3)
            else:
                i = copyI
            finally:
                continue
        elif k == ord('c'):
            k = 'c'
        elif k == ord('o'):
            k = 'o'
        elif k == ord('g'):
            k = 'g'
        elif k == ord('r'):
            k = 'r'
        elif k == ord('k'):
            k = 'k'
        elif k == ord('u'):
            if len(editingHistory) == 0:
                print('essa imagem nao foi editada')
                time.sleep(3)
            else:
                imgBeingEdited = editingHistory.pop()
                cv2.destroyAllWindows()
                cv2.imshow(f'{imgBeingEditedName}', imgBeingEdited)
            continue

        try:
            copyImg = imgBeingEdited.copy()
            imgBeingEdited = modificateImg[k](imgBeingEdited)
            editingHistory.push(copyImg)
            cv2.destroyAllWindows()
            cv2.imshow(f'{imgBeingEditedName}', imgBeingEdited)
        except KeyError:
            pass


# orchestratingTheMainFlux***************************************
logging.basicConfig(filename='logs.log',
                    filemode='w',
                    format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)

usersEncoding, knownUsersImgsPath = loadKnownUsers()
userName = authenticateUser(usersEncoding, knownUsersImgsPath)
app(userName)
