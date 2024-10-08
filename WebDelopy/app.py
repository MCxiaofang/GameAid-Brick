from flask import Flask, request, send_file, jsonify
import os
import io
import base64
from PIL import Image
import cv2
import numpy as np
from game_aid import hints, solution
import threading
import time

app = Flask(__name__, static_folder='static')

# 用于存储处理状态和结果的全局变量
processing_status = {
    'progress': 0,
    'searchSteps': 0,
    'images': [],
    "active_timestamp": 0,
    "thread": None,
    "stop_flag": False
}

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/hints', methods=['POST'])
def get_hints():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        image = Image.open(file.stream)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result_image_cv = hints(image_cv)
        result_image = Image.fromarray(cv2.cvtColor(result_image_cv, cv2.COLOR_BGR2RGB))
        img_io = io.BytesIO()
        result_image.save(img_io, 'JPEG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')

@app.route('/solution', methods=['POST'])
def start_solution():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        image = Image.open(file.stream)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 如果thread当前正在运行，则关闭上一个线程
        if processing_status["thread"] is not None:
            processing_status["stop_flag"] = True
            processing_status["thread"].join()
            print("Last thread stopped")

        processing_status['progress'] = 0
        processing_status['searchSteps'] = 0
        processing_status['images'] = []
        processing_status['active_timestamp'] = time.time()

        processing_status["thread"] = threading.Thread(target=process_solution, args=(image_cv,))
        processing_status["thread"].start()
        return "Processing started", 202

def process_solution(image_cv):
    result_images_cv = solution(image_cv, processing_status)
    
    result_images_base64 = []
    for result_image_cv in result_images_cv:
        result_image = Image.fromarray(cv2.cvtColor(result_image_cv, cv2.COLOR_BGR2RGB))
        img_io = io.BytesIO()
        result_image.save(img_io, 'PNG')
        img_io.seek(0)
        result_images_base64.append(base64.b64encode(img_io.getvalue()).decode('utf-8'))

    processing_status['images'] = result_images_base64
    processing_status['progress'] = 70
    processing_status['thread'] = None

@app.route('/solution_progress', methods=['GET'])
def get_solution_progress():
    processing_status['active_timestamp'] = time.time()
    return jsonify(progress=processing_status['progress'], searchSteps=processing_status['searchSteps'])

@app.route('/solution_result', methods=['GET'])
def get_solution_result():
    index = request.args.get('index', default=None, type=int)
    if index is not None and 0 <= index < len(processing_status['images']):
        return jsonify(image=processing_status['images'][index])
    else:
        return jsonify(error="Invalid index"), 400

@app.route('/solution_cnt', methods=['GET'])
def get_solution_count():
    return jsonify(count=len(processing_status['images']))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9001)
