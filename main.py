


from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, File, Body
import numpy as np
from starlette.requests import Request
from fastapi.responses import JSONResponse, Response
from fastapi.encoders import jsonable_encoder
import os
import cv2
import time
import datetime
import json
import base64
import io
from urllib.request import urlopen
from PIL import Image, ImageDraw, ImageColor, ImageFont
import boto3
import onnxruntime as rt
import boto3
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing import image
import requests
import textwrap

app = FastAPI()


def text_on_image(img, text, sale):

    h_red = cv2.calcHist([img], [2], None, [256], [0,256])
    h_green = cv2.calcHist([img], [1], None, [256], [0,256])
    h_blue = cv2.calcHist([img], [0], None, [256], [0,256])

    # Remove background pixels from the histograms.
    # Set histogram bins above 230 with zero 
    # assume all text has lower values of red, green and blue.
    h_red[230:] = 0
    h_green[230:] = 0
    h_blue[230:] = 0

    # Compute number of elements in histogram, after removing background
    count_red = h_red.sum()
    count_green = h_green.sum()
    count_blue = h_blue.sum()

    # Compute the sum of pixels in the original image according to histogram.
    # Example:
    # If h[100] = 10
    # Then there are 10 pixels with value 100 in the image.
    # The sum of the 10 pixels is 100*10.
    # The sum of an pixels in the original image is: h[0]*0 + h[1]*1 + h[2]*2...
    sum_red = np.sum(h_red * np.c_[0:256])
    sum_green = np.sum(h_green * np.c_[0:256])
    sum_blue = np.sum(h_blue * np.c_[0:256])

    # Compute the average - divide sum by count.
    avg_red = int(sum_red / count_red)
    avg_green = int(sum_green / count_green)
    avg_blue = int(sum_blue / count_blue)

    #print('Text RGB average is about: {}, {}, {}'.format(avg_red, avg_green, avg_blue))

    # font
    font = cv2.FONT_HERSHEY_DUPLEX 
    # fontScale
    fontScale = 1
    #text = "hello"
    thickness = 2
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    #print("textsize",textsize[0])
    #font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    height, width, channels = img.shape
    #print(height, width)
    x = int(0.1*width)
    y = int(0.1*height)
    #print(y)
    #print(x)
    org = (x, y)
    color = (255-avg_blue, 255-avg_green, 255-avg_red)
    color1 = (avg_blue, avg_green, avg_red)
    x1 = int(0.8*width)
    y1 = int(0.2*height)
    org1 = (x1,y1)
    font1 = cv2.FONT_HERSHEY_SIMPLEX
    textsize1 = cv2.getTextSize(sale, font1, fontScale, thickness)[0]
    c_y = y1 + textsize1[0]/2
    c_x = x1 + textsize1[1]/2
    CENTER = (int(c_x), int(c_y))
    text_origin = (CENTER[0] - textsize1[0] / 2, CENTER[1] + textsize1[1] / 2)
    text_orgin = (int(text_origin[0]),int(text_origin[1]))
    radius = int(textsize1[0]/2 + 10)
    cv2.circle(img, CENTER, radius, color, -1)
    cv2.putText(img, sale, text_orgin, font1, fontScale, color1, thickness, cv2.LINE_AA, False)
    
    text_string_len = int(width + 0.2*width)
    if int(textsize[0])+ int(0.1*width) > text_string_len:
        text_len = int(len(text)/2)
        #print(text_len)
        wrapped_text = textwrap.wrap(text, width=text_len)

    
        for i, line in enumerate(wrapped_text):
            textsize = cv2.getTextSize(line, font, fontScale, thickness)[0]

            gap = textsize[1] + 10

            y = int((img.shape[0] + textsize[1]) / 2) + i * gap
            x = int((img.shape[1] - textsize[0]) / 2)
            #print(x, y)
            cv2.putText(img, line, (x,y), font,
                        fontScale, 
                        color, 
                        thickness, 
                        lineType = cv2.LINE_AA) 
    else:
        cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
    
    return img



@app.post("/text_on_image")
def image_gen(request: Request,text : str, sale : str, userPhoto: Optional[bytes] = File(None), url: Optional[str] = Body(None) ):
    if url is not None: 
        image_r = requests.get(url)
        fetch_status = image_r.status_code
        if fetch_status == 200:
            image = image_r.content
            img_np = cv2.imdecode(np.asarray(bytearray(image), dtype=np.uint8), 1)
    elif userPhoto is not None:
        nparr = np.fromstring(userPhoto, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        return {"response":"Please provide url or image"}
    #text = 'hey'
    img = text_on_image(img_np, text, sale)
    cv2.imwrite("output.jpg", img)
    return True
if __name__ == '__main__':
    unvicorn.run(app, host = '121.0.0.1', port=8000)
