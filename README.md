# Hand Gesture Recognition Diploma Project
[English](README.md) | [Polski](README.pl.md)

This project was developed as part of the **Diploma Project** course in the 6th semester of Applied Computer Science at Warsaw University of Technology.
**Author:** Mateusz Tyl

## Table of Contents

* [Overview](#overview)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Project Structure](#project-structure)
* [Running the App](#running-the-app)
* [Usage](#usage)
* [Quick Start](#quick-start)

## Overview

This application enables you to:

* Collect hand image data and 21 landmark coordinates
* Train a neural network model on that data
* Perform real-time hand-gesture (letter) detection
* “Sign” text from a provided text file in a guided exercise mode

## Prerequisites

* **Python 3.10.11**
* venv (Python virtual environment)
* Operating system: Windows, Linux or macOS

## Installation

1. **Clone the repository**
   git clone [https://github.com/MTT1804/HandSignAI](https://github.com/MTT1804/HandSignAI)
   cd HandSignAI
2. **Create and activate a virtual environment**
   python -m venv venv

   # Windows

   venv\Scripts\activate

   # Linux / macOS

   source venv/bin/activate
3. **Install dependencies**
   pip install -r requirements.txt

## Project Structure

main/ – core application code

main.py – entry point that launches the GUI

app_main.py – main HandDataCollectorApp class, UI setup & event loop

gui_collect.py – “Data Collection” tab (camera preview, save snapshots & CSV)

gui_train.py – “Model Training” tab (train/test split, epochs, callbacks)

gui_detection.py – “Real-time Detection” tab (live gesture inference)

gui_text_detection.py – “Text Signing” tab (guided text-based practice)

gui_instructions.py – “Instructions” tab (built-in user guide)

detection.py – low-level MediaPipe + model inference routines

training.py – model‐building & training logic (TensorFlow/Keras)

utils.py – helper functions (keyboard disabling, logging, etc.)

locales.py – translation strings loader and lookup

images/ – captured hand snapshots, organized by label

data/ – data.csv: collected 21-landmark coordinates + label + index

models/ – saved Keras model file (.h5) and scaler pickle (.pkl)

other/ – application logs (logs.log) and miscellaneous outputs

text_files/ – sample .txt files used in the “Text Signing” tab

requirements.txt – Python dependencies

README.md – project overview, installation & usage instructions

## Running the App

1. **Change into the main folder**
   cd main
2. **Launch the application**
   python main.py

## Usage

When you launch the application, you’ll see five main tabs:

1. **Data Collection**
   Capture hand images and save 21 hand-landmark coordinates to `data/data.csv` and snapshots in `images/`.
2. **Model Training**
   Configure parameters (test size, random seed, epochs, batch size, patience) and train the neural network, saving the `.h5` model and `.pkl` scaler.
3. **Real-time Detection**
   Perform live gesture recognition—detected letters stream into the text pane, and a “Top-10” probabilities window shows confidence scores.
4. **Text Signing**
   Load a text file, then practice “signing” each character; correct signs turn green and statistics update live.
5. **Instructions**
   A full user guide is built into the app under the **Instructions** tab.

## Quick Start

Follow these four steps to go from data collection to real-time detection:

### 1. Collect Data

1. Launch the app and open the **Data Collection** tab.
2. Select your camera and enter the letter/number you wish to record.
3. Position your hand clearly in front of the camera (avoid shadows).
4. Click **Save Data** (or press **Enter**) to record the 21 landmarks to CSV and save a snapshot under `images/`.
5. Repeat for each class; collect at least **100–200 samples** per class at varying angles.

### 2. CSV Format

Your file `data/data.csv` will have **44 columns** per row:

* `x0, y0, x1, y1, …, x20, y20` — normalized landmark coordinates
* `label` — the letter or digit
* `index` — sample index

Verify the header to ensure all columns are present.

### 3. Train the Model

1. Go to the **Model Training** tab.
2. Point to your CSV, model output path (`.h5`) and scaler output path (`.pkl`).
3. Adjust parameters as needed:

   * **Test size**: 0.1–0.2
   * **Batch size**: 16–32
   * **Epochs**: 20–50
   * **Patience**: 5
4. Click **Start Training**. Progress runs in the background.
5. Once training completes, review test accuracy and find your saved model and scaler.

### 4. Real-time Detection

1. Switch to the **Real-time Detection** tab.
2. Ensure you have the correct model and scaler loaded.
3. Set your **interval** (e.g. 1000 ms) and **confidence threshold** (e.g. 0.7).
4. Click **Start Detection**—recognized letters will appear in the text box.
5. (Optional) Enable **“Insert only on Enter”** mode to manually confirm each prediction.

---

You’re all set! Collect, train, and test your own hand-gesture recognition pipeline. Enjoy!
