from transforms.Transform import *
import glob
import logging
import cv2

logging.basicConfig(level=logging.INFO)
images = "camera_cal/*.jpg"

imgNames = glob.glob(images)
# cameraCal = CameraCalibration(images, (9, 6))

# for i, imName in enumerate(imgNames):
#     img = cv2.imread(imName)
#     logging.info("Image {} size {}".format(imName, img.shape))
#     dst = cameraCal.transform(img)
#     outFilename = "output_images/" + imName
#     logging.info("Writing file {}".format(outFilename))
#     cv2.imwrite(outFilename, dst)

imgNames = glob.glob("test_images/*.jpg")
transform = Thresholding(kernel_size=5, sobel_thresh=(20, 100), s_thresh=(170, 255))
for i, imName in enumerate(imgNames):
    img = cv2.imread(imName)
    dst = transform.transform(img)
    dstFile = "output_images/thresholding/" + imName
    logging.info("Writing file {}".format(dstFile))
    cv2.imwrite(dstFile, dst)