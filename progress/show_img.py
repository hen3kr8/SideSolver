import matplotlib.pyplot as plt
import cv2
if __name__ == "__main__":
    image = cv2.imread('applied_homography.png')
    plt.imshow(image ,cmap='gray')
    plt.show()

