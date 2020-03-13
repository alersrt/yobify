import cv2
import numpy
from flask import Flask, request, make_response
from mtcnn import MTCNN

app = Flask(
    __name__,
    static_url_path='',
    static_folder='../resources/static'
)
detector = MTCNN()
yoba = cv2.cvtColor(cv2.imread('../resources/static/yoba.png', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGBA)


@app.route('/detect', methods=['POST'])
def detect():
  pixels = cv2.cvtColor(cv2.imdecode(numpy.frombuffer(request.files['file'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED),
                        cv2.COLOR_BGR2RGB)
  bboxes = detector.detect_faces(pixels)
  for bbox in bboxes:
    box, confidence, keypoints = bbox.values()
    left_eye, right_eye, nose, mouth_left, mouth_right = keypoints.values()
    x, y, width, height = box
    x2, y2 = x + width, y + height
    yo = cv2.resize(yoba, dsize=(width, height), interpolation=cv2.INTER_AREA)
    alpha_s = yo[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
      pixels[y:y2, x:x2, c] = (alpha_s * yo[:, :, c] + alpha_l * pixels[y:y2, x:x2, c])
  retval, buffer = cv2.imencode('.png', cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGRA))
  response = make_response(buffer.tobytes())
  response.headers['Content-Type'] = 'image/png'
  return response, 200


@app.route('/')
def index():
  return app.send_static_file(filename='index.html')


if __name__ == '__main__':
  app.run(debug=True)
