""" 最小限のサンプル
"""
import os
from PIL import Image
import cv2 as cv


def min_sample():
    # 画像をグレースケール化
    # グレースケール化しなくてもよいが、グレースケール化した方が高速とのこと
    img_file_name = 'input1.jpg'
    img_path = f'./input/{img_file_name}'
    img = cv.imread(img_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imwrite(f'./output/{img_file_name}_gray.jpg', img_gray)   # for test
    
    # 学習済みxmlはこちらから入手
    # https://github.com/opencv/opencv/tree/master/data/haarcascades
    xml_path = os.path.join(
        os.path.dirname(__file__),
        './lib/opencv/data/haarcascade_frontalface_default.xml'
    )
    # 分類器をインスタンス化
    face_cascade = cv.CascadeClassifier(xml_path)

    # 認識
    # scaleFactorは、デフォルトは1.1、1.0に近いほど細かくチェックし、見逃しは少なく誤検出が増える
    # minNeighborsは、デフォルトは3、0に近いほど見逃しは少なく誤検出が増える
    detected_faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.11,
        minNeighbars=3
    )
    
    # 認識した個所に矩形を書いて保存
    for i,(x,y,w,h) in enumerate(detected_faces):
        # 切り出して保存
        #cv.imwrite(f'./output/{img_file_name}_{i}.jpg', img[y-10:y+h+10, x-10:x+w+10])
        
        # 矩形を描画（opencvは、BGRの順）
        cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), thickness=5)
    # 保存
    cv.imwrite(f'./output/{img_file_name}', img)
        

if __name__=='__main__':
    min_sample()
