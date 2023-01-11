import numpy as np
import cv2
import pytesseract
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros
from itertools import product


class Tema2:

    def grayscale(self,image):

        # Transformam toti pixelii in pixeli gray
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray


    def gen_gaussian_kernel(self, k_size, sigma):
        # Calculam kernelul gaussian conform teoriei
        # Sursa(1): https://pages.stat.wisc.edu/~mchung/teaching/MIA/reading/diffusion.gaussian.kernel.pdf
        center = k_size // 2
        x, y = mgrid[0 - center : k_size - center, 0 - center : k_size - center]
        g = 1 / (2 * pi * sigma) * exp(-(square(x) + square(y)) / (2 * square(sigma)))
        return g


    def gaussian_filter(self, image, k_size, sigma):

        # Extragem inaltime si latimea imaginii
        height, width = image.shape[0], image.shape[1]
        
        # Calculam latimea si inaltime
        # Luand in considerare parametrul de nucleu furnizat
        dst_height = height - k_size + 1
        dst_width = width - k_size + 1

        # Construim imaginea conform teoriei
        image_array = zeros((dst_height * dst_width, k_size * k_size))
        row = 0
        for i, j in product(range(dst_height), range(dst_width)):
            window = ravel(image[i : i + k_size, j : j + k_size])
            image_array[row, :] = window
            row += 1

        # Transformam kernelul intr-o forma (k patrat, 1)
        gaussian_kernel = self.gen_gaussian_kernel(k_size, sigma)
        filter_array = ravel(gaussian_kernel)

        # Facem produsul celor 2: imagine si filtrul
        # Facem reshape
        # Transformam rezultatul in 8 canale
        dst = dot(image_array, filter_array).reshape(dst_height, dst_width).astype(uint8)

        # Returnam
        return dst

    def blur(self, image):
        image = self.gaussian_filter(image, 5, sigma=0.8)

        return image

    def sobel(self, image):
        # Aplicam pe imagine grayscale
        # Apoi ii aplicam un gaussian blur
        # Aceste 2 operatii pentru netezirea imaginii
        image = self.blur(self.grayscale(image))

        # Initializm o matrice de convolutie
        # de marimea imaginii, dar cu pixelii setati pe 0
        convolved = np.zeros(image.shape)

        # Initializam 2 matrici
        # Una petru orizontal, una pentru vertical
        # Sursa(1): https://en.wikipedia.org/wiki/Sobel_operator
        # Sursa(2): Lab SPG
        G_x = np.zeros(image.shape)
        G_y = np.zeros(image.shape)
        
        # Extragem tuplul cu marimea imaginii
        size = image.shape

        # Declaram la fel ca mai sus
        # 2 matrici orizontal, si vertical pentru kernenul filtrului 
        # Sursa(1): Lab SPG
        # Sursa(2): Implementare matlab https://en.wikipedia.org/wiki/Sobel_operator
        kernel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
        kernel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))


        # Completam matricile orizontale si verticale
        # Conform teoriei
        # https://en.wikipedia.org/wiki/Sobel_operator
        for i in range(1, size[0] - 1):
            for j in range(1, size[1] - 1):
                G_x[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], kernel_x))
                G_y[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], kernel_y))
        
        # Pentru a forma imaginea la loc
        # Vom folosi operatia de convolutie conform teoriei
        convolved = np.sqrt(np.square(G_x) + np.square(G_y))
        convolved = np.multiply(convolved, 255.0 / convolved.max())

        angles = np.rad2deg(np.arctan2(G_y, G_x))
        angles[angles < 0] += 180

        # Transformam imaginea in 8 canale
        convolved = convolved.astype('uint8')

        # Returnam imaginea si unghiurile
        return convolved, angles

    def nms(self, image, angles):
        # Extragem tuplul cu dimensiunile
        size = image.shape

        # Initializam o matrice de
        # dimensiunea imaginii
        suppressed = np.zeros(size)

        # Iteram prin pixelii imaginii
        # Si incercam sa gasim 
        for i in range(1, size[0] - 1):
            for j in range(1, size[1] - 1):
                if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                    value_to_compare = max(image[i, j - 1], image[i, j + 1])
                elif (22.5 <= angles[i, j] < 67.5):
                    value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
                elif (67.5 <= angles[i, j] < 112.5):
                    value_to_compare = max(image[i - 1, j], image[i + 1, j])
                else:
                    value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])
                
                if image[i, j] >= value_to_compare:
                    suppressed[i, j] = image[i, j]

        suppressed = np.multiply(suppressed, 255.0 / suppressed.max())


        return suppressed

    def double_threshold_hysteresis(self, image, low, high):

        # Definim tresholdul minim si maxim
        weak = 50
        strong = 255

        # Extragm tuplul cu dimensiunile imaginii
        size = image.shape

        # Initializam rezultatul ca fiind o matrice
        # De dimensiunea imaginii initiale
        result = np.zeros(size)

        # Extragem edge-urile care se incadreaza la weak si storng
        weak_x, weak_y = np.where((image > low) & (image <= high))
        strong_x, strong_y = np.where(image >= high)

        # Setam in matricea rezultat
        # Calculele anterioare
        result[strong_x, strong_y] = strong
        result[weak_x, weak_y] = weak

        # Definim directiile orizontale si verticale
        dx = np.array((-1, -1, 0, 1, 1, 1, 0, -1))
        dy = np.array((0, 1, 1, 1, 0, -1, -1, -1))
        size = image.shape
        
        # Calculam edge-urile care raman in imagine
        while len(strong_x):
            x = strong_x[0]
            y = strong_y[0]
            strong_x = np.delete(strong_x, 0)
            strong_y = np.delete(strong_y, 0)
            for direction in range(len(dx)):
                new_x = x + dx[direction]
                new_y = y + dy[direction]
                if((new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and (result[new_x, new_y]  == weak)):
                    result[new_x, new_y] = strong
                    np.append(strong_x, new_x)
                    np.append(strong_y, new_y)
        
        result[result != strong] = 0
        return result

    def canny(self, image, low, high):

        # Pentru a aplica filtrul CANNY
        # Trebuie sa urmam pasii de mai jos:

        # Aplicam sobel pe imagine
        image, angles = self.sobel(image)
        
        # Aplicam NMS (non-maxima supression)
        # Pentru a minimiza contururile ne-necesare
        image = self.nms(image, angles)
                
        # Aplicam Hysteresis thresholding
        # Pentru a netezi imaginea si a scoate in evidenta contururile
        image = self.double_threshold_hysteresis(image, low, high)
        
        return image


    # Paper: https://hackernoon.com/learn-k-means-clustering-by-quantizing-color-images-in-python
    # Reducerea culorilor folosind K-MEANS (BONUS)
    def reduce_colors(self,image, num_colors):

        # Convertim imaginea pentru cele 3 canale
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Initializam clustere cu centre random
        indices = np.random.randint(0, pixels.shape[0], size=num_colors)
        centroids = pixels[indices]

        # Repeteam algoritmul pana converge conform teoriei
        for _ in range(10):
            # Calculam pentru fiecare pixel distanta folosind formula din paper
            # Pentru fiecare pixel ii asignam cel mai apropait centru
            distances = np.sqrt(((pixels - centroids[:, np.newaxis])**2).sum(axis=2))
            closest_centroids = np.argmin(distances, axis=0)
            
            # Actualizam centrele
            for i in range(num_colors):
                points = pixels[closest_centroids == i]
                if len(points) != 0:
                    centroids[i] = points.mean(axis=0)

        # Inlocuim fiecare pixel cu centrul respectiv
        # Apoi convertim la 8canale
        new_pixels = centroids[closest_centroids]
        new_pixels = new_pixels.reshape(image.shape).astype(np.uint8)

        return new_pixels

    def inverse_image(self, image):
        im = np.array(image)

        mask = np.full(im.shape,255)

        mod_img = mask - im
        mod_img = mod_img.astype(np.uint8)

        return mod_img

    def combine_images(self, image1, image2):
        # Create an empty image for the blend
        blend = np.zeros([image1.shape[0], image1.shape[1], 3], dtype=np.uint8)

        # Iteram prin toti pixelii
        # Si daca avem in imaginea canny pixel negru
        # Il scriem
        # Daca nu, scriem pixelul din imaginea principala
        for y in range(image1.shape[0]):
            for x in range(image1.shape[1]):
                if(image1[y,x] == 0):
                    blend[y,x] = 0
                else:
                    blend[y,x] = image2[y,x]
        return blend


    def median_filter(self, data, filter_size):

        return data

    def solve_homework(self, image, show=True):

        # Generam imaginea folosind canny
        canny_image = self.canny(image, 0, 50)

        # Inversam culorile
        canny_image = self.inverse_image(canny_image)

        # Setam sa fie pe 8 canale
        # Ca sa arate bine pe imshow
        canny_image = canny_image.astype(np.uint8)

        reduced_image = self.reduce_colors(image, 16)

        # TODO: Filtru Median
        median_filter = self.median_filter(reduced_image, 3)

        combined_images = self.combine_images(canny_image, median_filter)

        if show:
            cv2.imshow("Image", image)
            cv2.waitKey(0)

            cv2.imshow("Combined", combined_images)
            cv2.waitKey(0)

            cv2.imshow("Canny Image", canny_image)
            cv2.waitKey(0)

            cv2.imshow("Reduced Image", reduced_image)
            cv2.waitKey(0)

        # Returnam si afisam numarul de inamtriculare
        return canny_image, reduced_image