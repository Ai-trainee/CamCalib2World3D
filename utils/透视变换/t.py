import cv2
import numpy as np

# Load the image
image = cv2.imread(r'78.png ')
if image is None:
    print("Error loading image!")
    exit()

height, width, _ = image.shape

# Define the 4 corners of the image
src_points = np.array([
    [273, 300.0],
    [354.0, 303],
    [269.0, 345],
    [356, 348.0]
], dtype=np.float32)


# Define the corresponding 4 points on the ground in world coordinates
dst_points = np.array([
    [0, 0],
    [640, 0],
    [0, 480],
    [640, 480]
], dtype=np.float32)

# Compute the perspective transformation matrix
M = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply the transformation
transformed_image = cv2.warpPerspective(image, M, (800, 600))

# Display the transformed image
cv2.imshow('Transformed Image', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# import cv2
# import numpy as np
#
#
# def perspective_transform(img_path, output_size):
#     # Read the image
#     img = cv2.imread(img_path)
#
#     # Get the dimensions of the image
#     h, w = img.shape[:2]
#
#     # Define the source points as the corners of the original image
#     src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
#
#     # Define the destination points as the corners of the desired output size
#     dst = np.float32([[0, 0], [output_size[0], 0], [0, output_size[1]], [output_size[0], output_size[1]]])
#
#     # Compute the perspective transform matrix
#     M = cv2.getPerspectiveTransform(src, dst)
#
#     # Apply the transformation
#     transformed_img = cv2.warpPerspective(img, M, output_size)
#
#     return transformed_img
#
#
# # Test the function
# output_size = (1000, 1800)  # Desired output size (width, height)
# img_path = r'D:\Desktop\Calibration-ZhangZhengyou-Method-master\pic\RGB_camera_calib_img\100001.png'
# result = perspective_transform(img_path, output_size)
#
# # Display the result
# cv2.imshow('Transformed Image', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
