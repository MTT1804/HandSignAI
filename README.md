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

* Collect hand image data and hand-landmark coordinates (up to 4 hands, 21 points each)
* Train a neural network model on that data
* Perform real-time hand-gesture (letter) detection
* Practice “signing” text from a provided text file in a guided exercise mode

## Prerequisites

* **Python 3.10.11**
* venv (Python virtual environment)
* Operating system: Windows, Linux or macOS

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MTT1804/HandSignAI.git
   cd HandSignAI
   ```
2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux / macOS
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

main/ – core application code and runtime data

main/main.py – entry point that launches the GUI

main/ctk_app/ – CustomTkinter application package

main/ctk_app/app.py – App controller (views, camera loop, training hooks)

main/ctk_app/views/ – UI views (data collection, detection, training, text practice, instructions, settings)

main/ctk_app/training_worker.py – model building & training logic (TensorFlow/Keras)

main/locales/ – translation strings and loader

main/images/ – captured hand snapshots, organized by label

main/data/ – data.csv: extracted features + label + index

main/models/ – saved Keras model file (.h5)

main/other/ – settings, scaler pickle (.pkl), logs, themes

main/text_files/ – sample .txt files used in the Text Practice tab

requirements.txt – Python dependencies

README.md – project overview, installation & usage instructions

## Running the App

1. **Change into the main folder**
   ```bash
   cd main
   ```
2. **Launch the application**
   ```bash
   python main.py
   ```

## Usage

When you launch the application, you’ll see six main tabs:

1. **Data Collection**
   Capture hand images and save extracted features to `main/data/data.csv` and snapshots in `main/images/`.
2. **Real-time Detection**
   Perform live gesture recognition—detected letters stream into the text pane, and a “Top-10” probabilities window shows confidence scores.
3. **Model Training**
   Configure parameters (test size, random seed, epochs, batch size, patience) and train the neural network, saving the `.h5` model and the `.pkl` scaler.
4. **Text Practice**
   Load a text file, then practice “signing” each character; correct signs are underlined and statistics update live.
5. **Instructions**
   A full user guide is built into the app under the **Instructions** tab.
6. **Settings**
   Configure paths, UI theme/language, detection options, and MediaPipe parameters.

## Quick Start

Follow these four steps to go from data collection to real-time detection:

### 1. Collect Data

1. Launch the app and open the **Data Collection** tab.
2. Select your camera and enter the letter/number you wish to record.
3. Position your hand clearly in front of the camera (avoid shadows).
4. Click **Save Data** (or press **Enter**) to record the 21 landmarks to CSV and save a snapshot under `main/images/`.
5. Repeat for each class; collect at least **100–200 samples** per class at varying angles.

### 2. CSV Format

Your file `main/data/data.csv` will contain one row per sample. By default it supports up to **4 hands** and stores:

* `h1_x0, h1_y0, …, h1_x20, h1_y20` — normalized landmarks for hand #1
* …
* `h4_x0, h4_y0, …, h4_x20, h4_y20` — normalized landmarks for hand #4
* `label` — the class (letter/digit/special)
* `index` — sample index

That is **170 columns** total with the default configuration (4 hands × 21 landmarks × 2 coords + label + index). Missing hands are padded with zeros.

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
