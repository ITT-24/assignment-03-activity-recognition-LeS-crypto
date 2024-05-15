## 01 - gather-data.py
- reads the sensor data from the DIPPID device until a sample limit is reached
- repeates for 5 iterations
- resamples the data to 100Hz before saving (see: resample.py)
- drops a couple of rows, but only like 2,3 per iteration
- restart for recording 1 activity 

## 02 - fitness-trainer.py
- press <kbd>Enter</kbd> after starting the application to train the model
  - uses the data from #01 
  - model is trained `activity_recognizer.py`
- possible exercises are shown using simple animations
- the predicted exercise will highlight the coresponding animations (full opacity)

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/6zlI_xU2)
