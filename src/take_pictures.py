import os
import cv2
import numpy as np
from picamera2 import Picamera2
from time import sleep

def write_image(output_folder: str, img_name: str, img: np.array):
    img_path = os.path.join(output_folder, f"{img_name}.jpg")
    cv2.imwrite(img_path, img)

def stream_video():
    picam = Picamera2()
    picam.preview_configuration.main.size=(960, 540)
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    n = 0

    while True:
        frame = picam.capture_array()

        cv2.imshow("picam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('p'):
            print("Guardando foto ...")
            write_image(f"../data/calibration", f"{n}", frame)
            n += 1
            sleep(5)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_video()