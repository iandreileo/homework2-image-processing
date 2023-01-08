import cv2
import argparse
from Tema2 import Tema2
import time


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())

    # Citim path-ul imaginii de la tastatura
    image = cv2.imread(args["image"])

    # Dam start la timer
    # Ca sa testam cat dureaza algoritmul
    start = time.time()

    # Aplicam algoritmul pe poza
    canny = Tema2()

    imagine_finala = canny.solve_homework(image, True)

    # Verificam cat timp a durat algoritmul
    end = time.time()
    print(end - start)

