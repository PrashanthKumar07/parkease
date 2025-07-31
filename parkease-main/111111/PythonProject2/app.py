from flask import Flask, render_template, redirect, url_for, request, session, flash
import cv2
import imutils
import numpy as np
from datetime import datetime
import csv
import os
import time
import easyocr  # OCR alternative
import re
from functools import wraps

from pandas import read_csv

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure key

# --- ADMIN CREDENTIALS (for demo purposes) ---
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'password123'

# Global variable for the last detected plate
last_detected_plate = None

# Initialize EasyOCR Reader for English (using CPU)
reader = easyocr.Reader(['en'], gpu=False)

# Define folders for static files
IMAGE_FOLDER = os.path.join("static", "images")
DATA_FOLDER = os.path.join("static", "data")

# Create directories if they don't exist
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# CSV file paths (data stored in static/data/)
ACTIVE_CSV = os.path.join(DATA_FOLDER, "active.csv")
HISTORY_CSV = os.path.join(DATA_FOLDER, "history.csv")

# Define cost rate per minute (for example, 5 currency units per minute)
RATE_PER_MINUTE = 0.5
# Ensure CSV files exist; if not, create them with headers.
if not os.path.exists(ACTIVE_CSV):
    with open(ACTIVE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plate_number", "entry_time"])

if not os.path.exists(HISTORY_CSV):
    with open(HISTORY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plate_number", "entry_time", "exit_time", "duration", "cost"])

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash("Please log in to access this page.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def process_license_plate():
    cap = cv2.VideoCapture(0)
    detected_plate = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            break

        # Increase frame size if desired; example using 1280x720 (HD)
        frame = cv2.resize(frame, (1280, 720))

        # Preprocess the frame
        img = frame  # already resized
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 15, 75, 75)
        edged = cv2.Canny(bfilter, 30, 150)

        # Find contours and sort by area
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        screenCnt = None
        for c in contours:
            approx = cv2.approxPolyDP(c, 15, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is not None:
            cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask,[screenCnt],0,255,-1)
            new_image = cv2.bitwise_and(img, img, mask=mask)

            # Crop the detected region
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

            # Use EasyOCR to extract text from the cropped image
            result = reader.readtext(cropped)
            read = "".join([text for (_, text, _) in result])

            # Convert to uppercase and clean up
            read = read.upper()
            print("Uppercase OCR result:", read)
            read = ''.join(e for e in read if e.isalnum())

            # Remove unwanted country codes
            ignore_list = ["IND", "USA", "UK", "AUS", "CAN", "DEU", "FRA", "ITA", "ESP", "CHN", "JPN", "KOR"]
            for country in ignore_list:
                read = read.replace(country, "")
            print("Modified OCR result:", read)

            # Use regex to match
            # typical Indian plate pattern
            pattern = r"[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}"
            match = re.search(pattern, read)
            if match:
                detected_plate = match.group()
                print("Regex matched plate:", detected_plate)
            else:
                if 6 <= len(read) <= 11:
                    detected_plate = read
                else:
                    print("Detected plate did not meet length criteria after filtering.")
                    detected_plate = None

            if detected_plate:
                global last_detected_plate
                last_detected_plate = detected_plate

                process_vehicle_event(detected_plate)
                # Save processed images for debugging in the static/images folder
                cv2.imwrite(os.path.join(IMAGE_FOLDER, "debug_cropped.jpg"), cropped)
                cv2.imwrite(os.path.join(IMAGE_FOLDER, "debug_detection.jpg"), img)
                time.sleep(3)
                break
        else:
            print("No contour detected")
        # For testing, break after one iteration (remove for continuous scanning)
        break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error as e:
        print("cv2.destroyAllWindows() failed:", e)
    return detected_plate

def process_vehicle_event(plate):
    plate = plate.upper()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Read active.csv
    active_records = []
    with open(ACTIVE_CSV, "r", newline="") as f:
        reader_csv = csv.reader(f)
        active_records = list(reader_csv)

    # Skip header; find if plate exists
    record = None
    for row in active_records[1:]:
        if row[0] == plate:
            record = row
            break

    if record is None:
        # New entry: add record to active.csv
        with open(ACTIVE_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([plate, current_time])
        print(f"Vehicle {plate} entered at {current_time}")
    else:
        # Vehicle exit: calculate duration, cost, and update history.csv
        entry_time_str = record[1]
        entry_time_dt = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
        exit_time_dt = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
        duration = int((exit_time_dt - entry_time_dt).total_seconds() / 60)  # in minutes

        cost = duration * RATE_PER_MINUTE

        # Append to history.csv
        with open(HISTORY_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([plate, entry_time_str, current_time, duration, cost])
        print(f"Vehicle {plate} exited at {current_time} (Parked for {duration} minutes, Cost: {cost})")

        # Remove record from active.csv (rewrite file without this record)
        new_active = [row for row in active_records if row[0] != plate]
        with open(ACTIVE_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(new_active)

# --- Routes ---
@app.route('/')
def index():
    # For demonstration, show the most recent history record if available.
    last_record = None
    if os.path.exists(HISTORY_CSV):
        with open(HISTORY_CSV, "r", newline="") as f:
            reader_csv = csv.reader(f)
            records = list(reader_csv)
            if len(records) > 1:
                last_record = records[-1]

    global last_detected_plate
    plate_to_display = last_detected_plate

    return render_template('index.html', last_record=last_record, plate_to_display=plate_to_display)

@app.route('/scan', methods=['POST'])
def scan():
    detected_plate = process_license_plate()
    return redirect(url_for('index'))

# --- Login & Logout Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('user')
        password = request.form.get('pass')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            flash("Logged in successfully!")
            return redirect(url_for('admin_page'))
        else:
            flash("Invalid credentials, please try again.")
            return redirect(url_for('login'))
    return render_template('authentication.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash("Logged out.")
    return redirect(url_for('index'))


@app.route('/admin')
def adminpage():
    active_data = read_csv(ACTIVE_CSV)
    history_data = read_csv(HISTORY_CSV)

    return render_template("adminpage.html", history_data=history_data)

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/homepage')
def homepage():
    return render_template('homepage.html')

@app.route('/status')
def status_page():
    return render_template('statuspage.html')

@app.route('/updatebalance')
def update_balance_page():
    return render_template('updatebalance.html')

# New route to display parking history from CSV as a table
@app.route('/history')
def history():
    history_records = []
    if os.path.exists(HISTORY_CSV):
        with open(HISTORY_CSV, "r", newline="") as f:
            reader_csv = csv.reader(f)
            # Skip header row
            next(reader_csv, None)
            history_records = list(reader_csv)
    return render_template('history.html', history=history_records)

if __name__ == '__main__':
    app.run(debug=True)
