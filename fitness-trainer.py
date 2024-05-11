# this program visualizes activities with pyglet

import activity_recognizer as activity
import pyglet
import pandas as pd
import numpy as np
from sklearn import *
from time import sleep
from DIPPID import SensorUDP

"""
- [ ] (1P) the program correctly loads training data
- [ ] (2P) training data is pre-processed appropriately
- [ ] (1P) a classifier is trained with this training data when the program is started
- [ ] (3P) the classifier recognizes activities correctly
- [ ] (1P) prediction accuracy for a test data set is printed
- [ ] (1P) prediction works continuously without requiring intervention by the user
- [ ] (1P) the fitness training application works and looks nice
"""

# TODO:
    # display activities and track if executed correctly for x-amount of time
    # in activity_recognition.py
        # Program start = read training data from csv files
            # preprocess data
            # split into train(80) & test(20)
            # train machine learning classifier
            # After Training: evaluate model's accuracy using the test data set
        # Model should predict activities based on sensor data from DIPPID
    # Prediction should run continously without requiring further intervention by user
    # Visualize the fitness trainer using pyglet


# create window
# train model -> save classifier for later
# get DIPPID input -> if trained
# send input to activity_recognizer + get prediction back
# display results

PORT = 5700
sensor = SensorUDP(PORT)

WIDTH = 500
HEIGHT = 500
window = pyglet.window.Window(WIDTH, HEIGHT)
settings = pyglet.text.Label("Press Enter to train the model", x=WIDTH/2, y=HEIGHT/2, anchor_x='center', anchor_y='center')

# TODO: load exercize img and display them
# https://pyglet.readthedocs.io/en/latest/programming_guide/image.html


class Trainer():

    def __init__(self) -> None:
        self.finished_training = False
        self.prediction = pyglet.text.Label("Exercise: ...", x=WIDTH/2, y=HEIGHT/1.2, anchor_x='center', anchor_y='center' )

    def init_model(self):
        self.finished_training = activity.train_model()

    def init_activities(self):
        pass

    def read_input_stream(self):
        if not sensor.has_capability('accelerometer'):
            self.prediction.text = "No DIPPID device found"
        else:
            acc  = sensor.get_value('accelerometer')
            gyro = sensor.get_value('gyroscope')
            t = pd.Timestamp("1970-01-01") // pd.Timedelta('1ms')

            data = [acc['x'], acc['y'], acc['z'], gyro['x'], gyro['y'], gyro['z']]
            print(t, data)
            activity.use_model(data)

trainer = Trainer()

# ----- WINDOW INTERACTION ----- #

@window.event
def on_draw():
    window.clear()
    if trainer.finished_training:
        trainer.prediction.draw()
        trainer.read_input_stream()
    else:
        settings.draw()

@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.ESCAPE:
        window.close()
    elif symbol == pyglet.window.key.ENTER:
        settings.text = "Loading..."
        settings.draw() # doesn't work
        trainer.init_model()


if __name__ == "__main__":
    pyglet.app.run()


    # finished_training = activity.train_model()
    # if finished_training:
    #     print("has trained")
    #     # start rest