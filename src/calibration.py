import cv2
import numpy as np
import os

def load_images(filenames):
    return [cv2.imread(filename) for filename in filenames]

def show_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def write_image(path, image):
    cv2.imwrite(path, image)

def get_chessboard_points(chessboard_shape, dx, dy):
    return np.array([[(i%chessboard_shape[0])*dx, (i//chessboard_shape[0])*dy, 0] for i in range(chessboard_shape[0]*chessboard_shape[1])], dtype=np.float32)

chessboard_shape = (9, 6)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

imgs_path = [f"../data/calibration/{file}" for file in os.listdir("../data/calibration")]
imgs = load_images(imgs_path)

corners = [cv2.findChessboardCorners(img, chessboard_shape) for img in imgs]

imgs_gray = [cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in imgs]
corners_refined = [cv2.cornerSubPix(i, cor[1], chessboard_shape, (-1, -1), criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners)]

imgs_with_corners = [cv2.drawChessboardCorners(img, chessboard_shape, corners_refined_img, True) 
                     for corners_refined_img, img in zip(corners_refined, imgs) 
                     if len(corners_refined_img) > 0 ]

chessboard_points = [get_chessboard_points(chessboard_shape, 30, 30) for _ in imgs_with_corners]

valid_corners = np.array([cor[1] for cor in corners if cor[0]], dtype=np.float32)

valid_object_points = np.array(chessboard_points, dtype=np.float32)
image_points = np.array(valid_corners, dtype=np.float32)

rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(valid_object_points, image_points, imgs[1].shape[0:2], None, None)

extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

print("Intrinsics:\n", intrinsics)
print("Distortion coefficients:\n", dist_coeffs)
print("Root mean squared reprojection error:\n", rms)

os.makedirs("../out", exist_ok=True)
os.makedirs("../out/calibration", exist_ok=True)
for i, img in enumerate(imgs_with_corners):
    write_image(f"../out/calibration/{str(i).zfill(3)}.jpg", img)