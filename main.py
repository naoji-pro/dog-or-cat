from flask import Flask, request, redirect, render_template, flash
from flaskr  import app
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
import os
import numpy as np
from PIL import Image

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

#rsplitは'.'の右側1行を取得する
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
         
        if 'file' not in request.files:
                ans = 'ファイルありません'
                return render_template("index.html",answer=ans)
        
        file = request.files['file']
        
        if file.filename == '':
            ans = 'ファイルが空です'
            return render_template("index.html",answer=ans)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # アップロードされた画像を取得
            image = request.files['file']
            img = Image.open(image)
            
            # 画像をモデルに入力できる形式に変換
            img=img.resize((64,64))
            #img.show()
            model=load_model("./flaskr/model.h5")
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            result = model.predict(img_array)
            if result[0][0]>result[0][1]:         #（一次元目の）左の数字が大きいとき、猫と識別される。
                pred_answer = "これは猫です"
                num=100*(result[0][0])
                percent=str('{:.0f}'.format(num))+"%の確率でそのように判断しました。"
                return render_template("index.html",answer=pred_answer,probability=percent)
            else:
                pred_answer = "これは犬です"
                num=100*(result[0][1])
                percent=str('{:.0f}'.format(num))+"%の確率でそのように判断しました。"
                return render_template("index.html",answer=pred_answer,probability=percent)
        
    return render_template("index.html",answer="判定受け付け待ち")

if __name__ == "__main__":
    app.run()