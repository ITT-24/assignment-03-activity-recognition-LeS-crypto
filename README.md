## 01 - gather-data.py
- reads the sensor data from the DIPPID device until a sample limit is reached
  - repeates for 5 iterations
- resamples the data to 100Hz before saving (see: `resample.py`)
- drops a couple of rows, but only like 2,3 per iteration
- restart for recording one activity
- current data set was recorded with m5Stack held with the screen facing the side <small>(if that makes any difference)</small>

## 02 - fitness-trainer.py
- press <kbd>Enter</kbd> after starting the application to train the model
  - uses the data from #01 
  - model is trained with `activity_recognizer.py`
  - prints Accuracy Score, Classification Report and Confusion Matrix to the console
- possible exercises are shown using simple animations
- the predicted exercise will highlight the coresponding animations (i.e. full opacity)

#### Exercises:
- rowing: slow and full movement
- lifting: slow movent
- running: fast
- jumpingjacks: normal (detection not that good)

<small> activity_recognizer_test.py was for testing and some referencing, can be ignored </small>

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/6zlI_xU2)
