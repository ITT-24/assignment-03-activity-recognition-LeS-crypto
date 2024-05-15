# this program visualizes activities with pyglet

import activity_recognizer_3 as activity
import pyglet
import pandas as pd
import numpy as np
from sklearn import *
from time import sleep
import datetime
from DIPPID import SensorUDP
from pathlib import Path

"""
- [x] (1P) the program correctly loads training data
- [?] (2P) training data is pre-processed appropriately
- [x] (1P) a classifier is trained with this training data when the program is started
- [ ] (3P) the classifier recognizes activities correctly
- [x] (1P) prediction accuracy for a test data set is printed
- [x] (1P) prediction works continuously without requiring intervention by the user
- [/] (1P) the fitness training application works and looks nice
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
pyglet.gl.glClearColor(0.5,0,0,1)

settings = pyglet.text.Label("Press Enter to train the model", x=int(WIDTH/2), y=int(HEIGHT/2), 
                             anchor_x='center', anchor_y='center')

# LAYOUT & STYLING
OFF_SET = 130
MARGIN = 50
OPAC_NO = 128
OPAC_YES = 255

# THRESHOLDS
DATA_THRESHOLD = 200 # amount of data to collect before predicting
COUNTER_THRESHOLD = 1 # amount of times, the exercise has to be the same # for testing
ACC_CHANGE = 0.05 # change in accuracy to be considered significant movement

class Trainer():

    def __init__(self) -> None:
        self.finished_training = False
        self.prediction = pyglet.text.Label("Exercise: ...", x=int(WIDTH/2), y=int(HEIGHT/2), 
                                            anchor_x='center', anchor_y='center' )
        self.sensor_data = []
        self.previous_exercise = "..."
        self.exercise_counter = 0
        self.images = []

    def init_model(self):
        """Train and evaluate the model"""
        self.finished_training = activity.train_model()

    def read_input_stream(self):
        """Reads the DIPPID input stream and sends the collected data to the trained model"""

        if not sensor.has_capability('accelerometer'):
            self.prediction.text = "No DIPPID device found"
        else:
            # collect multiple sampless to be classified at once
            if len(self.sensor_data) > DATA_THRESHOLD:
                print("predicting activity...")
                act = activity.use_model(self.sensor_data)
                # print(act)
                self.sensor_data = []
                self.parse_prediction(act)
            else:
                # handle idle

                # TODO if sensor moves a significant amount -> should be handled in a_r
                acc  = sensor.get_value('accelerometer')
                gyro = sensor.get_value('gyroscope')
                t = datetime.datetime.now() # pd.Timestamp("1970-01-01") // pd.Timedelta('1ms')
                # t = (datetime.datetime.now() - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')

                data = [t, acc['x'], acc['y'], acc['z'], gyro['x'], gyro['y'], gyro['z']] # type: ignore
                if self.check_movement(data=data):
                    self.sensor_data.append(data)
                # print(t, data)
            # activity.use_model(data)
    
    def check_movement(self, data) -> bool:
        """Check if the DIPPID device moves a significant amount, i.e. isn't laying on the desk"""
        # Only check accerlerometer, as it seems most reliable => acc change </> 0.05
        if len(self.sensor_data) == 0:
            print("init")
            return True
        elif len(self.sensor_data) > 0: 
            # only check 1 -> all acc axis change when you move the device
            prev_acc_x = self.sensor_data[0][1]
            acc_x =  data[1]
            if acc_x > (prev_acc_x + ACC_CHANGE) or acc_x < (prev_acc_x - ACC_CHANGE):
                return True
        return False

            


    def parse_prediction(self, exercise:str):
        """ Parse and display the prediction.
        ?? Check if the predicted exercise has been performed for a certain amount"""
        print("detected")

        # print exercise name
        self.prediction.text = f"Excercise: {exercise}"
        self.prediction.draw()

        # change opacity of sprite
        images.select_action(exercise)
        
        # print(self.previous_exercise, "->", exercise)

        # if self.exercise_counter > COUNTER_THRESHOLD:
        #     print("detected")

        #     # print exercise name
        #     self.prediction.text = f"Excercise: {exercise}"
        #     self.prediction.draw()

        #     # change opacity of sprite
        #     images.select_action(exercise)

        # elif exercise == self.previous_exercise:
        #     print("same exercise")
        #     self.exercise_counter += 1

        # else: 
        #     print("new exercise ->", exercise)
        #     self.exercise_counter = 0
        #     self.prediction.text = f"Excercise: ..."
        #     self.prediction.draw()

        # self.previous_exercise = exercise


class Images():
    """Encompases all animated images that show the exercises"""

    actions = ["jumpingjacks", "lifting", "rowing", "running"]
    coordinates = [[MARGIN, MARGIN], [WIDTH-OFF_SET, MARGIN], 
                   [MARGIN, HEIGHT-OFF_SET], [WIDTH-OFF_SET, HEIGHT-OFF_SET]] # same order as actions
    
    def __init__(self) -> None:
        self.loader = pyglet.resource.Loader(['img'])
        self.batch = pyglet.graphics.Batch()
        self.animations = {}
        self.load_animations()

    def load_animations(self):
        """Load the images and create animation sprites"""
        for i in range(0, len(Images.actions)):
            key = Images.actions[i]
            img_1 = self.loader.image(f"{key}_1.png")
            img_2 = self.loader.image(f"{key}_2.png")
            img_1.width /= 10
            img_2.width /= 10
            img_1.height /= 10
            img_2.height /= 10

            frame_1 = pyglet.image.AnimationFrame(img_1, duration=0.5)
            frame_2 = pyglet.image.AnimationFrame(img_2, duration=0.5)
            ani = pyglet.image.Animation(frames=[frame_1, frame_2])

            pos = Images.coordinates[i]
            sprite = pyglet.sprite.Sprite(img=ani, x=pos[0], y=pos[1], batch=self.batch)
            sprite.opacity = OPAC_NO

            self.animations[key] = sprite
        print(self.animations)


    def select_action(self, act):
        """Change opacity of image to show a predicted action"""
        for key in self.animations:
            self.animations[key].opacity = OPAC_NO
        self.animations[act].opacity = OPAC_YES

# INIT
trainer = Trainer()
images = Images()

# ----- WINDOW INTERACTION ----- #

@window.event
def on_draw():
    window.clear()
    if trainer.finished_training:
        images.batch.draw()
        trainer.prediction.draw()
        trainer.read_input_stream()
    else:
        settings.draw()
        # print(images.sprite)
        # images.sprite.draw()
        # draw images

@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.ESCAPE:
        window.close()
    elif symbol == pyglet.window.key.ENTER:
        trainer.init_model()
        # settings.text = "Loading..."
        # settings.draw() # doesn't work
    # elif symbol == pyglet.window.key.Z:
    #     print("run")
    #     images.select_action("running")


if __name__ == "__main__":
    pyglet.app.run()


    # finished_training = activity.train_model()
    # if finished_training:
    #     print("has trained")
    #     # start rest