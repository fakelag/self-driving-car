import socketio
import eventlet
import base64
import model as mdl
import numpy as np

from io import BytesIO
from PIL import Image
from flask import Flask

# Setup sockets to connect to the simulator
sio = socketio.Server()
app = Flask(__name__)

# Load the model
model = mdl.model_load()

def send_control(steering_angle, throttle):
	print("steering: " + str(steering_angle))
	sio.emit("steer", data={
		"steering_angle": str(steering_angle),
		"throttle": str(throttle),
	})

@sio.on("connect")
def connect(sid, environ):
	send_control(0, 0)

@sio.on("telemetry")
def telemetry(sid, data):
	# A message was received
	# Decode the image and send back a steering prediction
	image = Image.open(BytesIO(base64.b64decode(data["image"])))
	image = np.asarray(image)
	image = mdl.image_process(image)
	image = np.array([image])
	steering_angle = float(model.predict(image))
	send_control(steering_angle, 1.0)

if __name__ == "__main__":
	app = socketio.Middleware(sio, app)
	eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
