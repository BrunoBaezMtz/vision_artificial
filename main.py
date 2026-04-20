# importar las librerias
from ultralytics import YOLO
import cv2

#leer nuestro modelo
modelo = YOLO ("best.pt")

# realizar videocaptura
cap =  cv2.VideoCapture(0)

#bucle
while True:
    # leer fotogramas
    ret, frame = cap.read()

    #leemos resultados
    resultados = modelo.predict(frame, imgsz = 640)

    # mostrar resultados
    anotaciones = resultados[0].plot()


    #mostrar nuestros fotogramas
    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)

    #cerrar programa
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyWindow()
