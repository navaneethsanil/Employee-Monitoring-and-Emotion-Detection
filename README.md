# Employee Monitoring and Emotion Detection

🎯 **Objective**  
A prototype web application that monitors employee presence via webcam, detects their emotional state, stores the data in a database, and displays results and analytics on a web dashboard.


## Features

1. **Camera Monitoring**
   - Uses OpenCV to detect employee presence via the system webcam.
   - Marks status as **Present** or **Not Present** in real-time.

2. **Emotion Detection**
   - Implements a deep learning model "SiglipForImageClassification" to recognize emotions.
   - Detects the following emotions: **Happy, Sad, Neutral, Angry, Surprise**.

3. **Backend & Database**
   - Stores logs in a database (SQLite by default; can be changed to MySQL/PostgreSQL).

4. **Dashboard (Frontend)**
   - Simple web dashboard using **HTML, CCS and Javascript**.
   - Displays:
     - Live camera preview.
     - Real-time detection results (Presence + Emotion).
     - Analytics: Presence timeline, Emotion distribution charts.

5. **Reports & Analytics**
   - Daily/weekly summaries:
     - Percentage of time present.
     - Most frequent emotion.


## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/navaneethsanil/Employee-Monitoring-and-Emotion-Detection.git
cd Employee-Monitoring-and-Emotion-Detection
```

### Create virtual environment and activate
```bash
python -m venv env
# Windows
env\Scripts\activate
# macOS/Linux
source env/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Apply database migrations
```bash
python manage.py migrate
```

### Run the Django server
```bash
python manage.py runserver
```


## Usage
* Open your browser and navigate to http://127.0.0.1:8000/.
* Allow access to your webcam.
* Monitor live employee presence and emotions in real-time.
* Check dashboard analytics for summaries and reports.
