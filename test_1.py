import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


import cv2
a = 0
# 1.creating a video object
while (a == 0):
    video = cv2.VideoCapture(0) 
    # 2. Variable
    a = 0
    # 3. While loop
    while True:
        a = a + 1
        # 4.Create a frame object
        check, frame = video.read()
        # Converting to grayscale
        #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # 5.show the frame!
        cv2.imshow("Capturing",frame)
        # 6.for playing 
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    # 7. image saving
    # 8. shutdown the camera
    video.release()
    cv2.destroyAllWindows 


    cv = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
    im = Image.fromarray(cv)



    # Replace this with the path to your image


    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(im, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array



    # run the inference
    prediction = model.predict(data)
    print(prediction)
    var = max(prediction[0])
    print(var)
    ar = np.array(prediction)
    lis1 = np.where(ar == var)
    print(lis1)
    if lis1[1] == 0:
        print("CLASS 1")
    elif lis1[1] == 1:
        print("CLASS 2")
    elif lis1[1] == 2:
        print("CLASS 3")
    elif lis1[1] == 3:
        print("CLASS 4")
    else :
        print("INVALID")
    flag = input("Do you want to continue?")
    if flag == 1:
        a += 1
