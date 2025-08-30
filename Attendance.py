import cv2
import pickle
import numpy as np
import os
import pandas as pd
import face_recognition
from datetime import datetime

def cleanup_previous_entries_for_today():
    """
    Reads the attendance file and removes any entries for the current day,
    while preserving manually added blank lines from previous sessions.
    """
    attendance_file = 'attendance.csv'
    today_str = datetime.now().strftime('%d-%m-%Y')
    lines_to_keep = []

    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            f.writelines('Name,Time,Date,Status\n')
        return

    with open(attendance_file, 'r') as f:
        lines = f.readlines()
        
    
    if lines:
        lines_to_keep.append(lines[0])
    
    # Process the rest of the lines
    for line in lines[1:]:
        # --- MODIFIED LOGIC ---
        # If the line is blank or just whitespace, keep it.
        if not line.strip():
            lines_to_keep.append(line)
            continue # Move to the next line

        # If the line has content, check its date.
        entry = line.split(',')
        if len(entry) > 2 and entry[2].strip() != today_str:
            lines_to_keep.append(line)

    # Write the filtered lines back to the file
    with open(attendance_file, 'w') as f:
        f.writelines(lines_to_keep)
    print(f"Cleaned up previous entries for {today_str}.")

# Load the trained face encodings and names.
print("-> Loading trained model...")
with open('encodings.pickle', 'rb') as f:
    data = pickle.load(f)
known_encodings = data['encodings']
known_names = data['names']
print("-> Model loaded successfully.")

# --- ADD THIS LINE ---
cleanup_previous_entries_for_today()

# ... (the rest of your script follows)
# Define cutoff times
GRACE_PERIOD_END = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
ATTENDANCE_CUTOFF = datetime.now().replace(hour=16, minute=30, second=0, microsecond=0)

# Load the trained face encodings and names.
print("-> Loading trained model...")
with open('encodings.pickle', 'rb') as f:
    data = pickle.load(f)
known_encodings = data['encodings']
known_names = data['names']
print("-> Model loaded successfully.")


def updatePercentage(name, status):
    """
    Updates attendance percentage. -1% for Late, -5% for Absent.
    """
    percentage_file = 'percentages.csv'
    if not os.path.exists(percentage_file):
        with open(percentage_file, 'w') as f:
            f.writelines('Name,Percentage\n')
            for student_name in set(known_names):
                f.writelines(f'{student_name},100\n')

    df = pd.read_csv(percentage_file)

    if name not in df['Name'].values:
        new_student = pd.DataFrame([{'Name': name, 'Percentage': 100}])
        df = pd.concat([df, new_student], ignore_index=True)
        print(f"New student '{name}' added to percentage tracker.")
    
    current_percentage = df.loc[df['Name'] == name, 'Percentage'].values[0]
    
    # --- MODIFIED: Apply different penalties based on status ---
    if status == 'Late':
        new_percentage = current_percentage - 1
        print(f"Percentage for {name} updated to {new_percentage}% (Late)")
    elif status == 'Absent':
        new_percentage = current_percentage - 3
        print(f"Percentage for {name} updated to {new_percentage}% (Absent)")
    else: # On-Time
        new_percentage = current_percentage
    
    df.loc[df['Name'] == name, 'Percentage'] = new_percentage
    df.to_csv(percentage_file, index=False)


def markAttendance(name):
    """
    Marks attendance in a CSV file with a status, and updates percentage.
    """
    with open('attendance.csv', 'a+') as f:
        f.seek(0)
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            if line.strip():
                entry = line.split(',')
                if len(entry) > 2:
                    nameList.append((entry[0], entry[2].strip()))

        today = datetime.now().strftime('%d-%m-%Y')
        if (name, today) not in nameList:
            now = datetime.now()
            status = "Late" if now >= GRACE_PERIOD_END else "On-Time"
            
            dtString = now.strftime('%H:%M:%S')
            dateString = now.strftime('%d-%m-%Y')
            
            f.writelines(f'\n{name},{dtString},{dateString},{status}')
            print(f"Attendance marked for {name} ({status})")
            updatePercentage(name, status)

# --- NEW: Function to mark absent students at the end ---
def markAbsentees(all_students, present_students):
    """
    Finds absent students and logs them to the CSV and percentage files.
    """
    absent_students = set(all_students) - set(present_students)
    print(f"\n--- Checking for Absentees ---")
    
    if not absent_students:
        print("All registered students are present.")
        return

    today = datetime.now().strftime('%d-%m-%Y')
    with open('attendance.csv', 'a') as f:
        for name in absent_students:
            f.writelines(f'\n{name},N/A,{today},Absent')
            print(f"Marking {name} as Absent.")
            updatePercentage(name, "Absent")


# Initialize video capture
video = cv2.VideoCapture(0)

while True:
    success, frame = video.read()
    if not success:
        print("Failed to capture frame.")
        break
        
    if datetime.now() >= ATTENDANCE_CUTOFF:
        cv2.putText(frame, "ATTENDANCE CLOSED", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Smart Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    small_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_loc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        match_index = np.argmin(face_distances)
        name = "Unknown"

        if matches[match_index]:
            if face_distances[match_index] < 0.5:
                name = known_names[match_index]
                markAttendance(name)

        top, right, bottom, left = face_loc
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name.title(), (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow("Smart Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- NEW: Logic to find and mark absentees after the loop ends ---
print("\nAttendance session has ended. Finalizing records...")
# 1. Get a unique list of all students registered in the model
all_registered_students = set(known_names)

# 2. Get the list of students who were marked present today
present_today = []
today_str = datetime.now().strftime('%d-%m-%Y')
if os.path.exists('attendance.csv'):
    with open('attendance.csv', 'r') as f:
        for line in f.readlines():
            if line.strip():
                entry = line.split(',')
                # Check if the entry is for today and the status is not 'Absent'
                if len(entry) > 3 and entry[2].strip() == today_str and entry[3].strip() != 'Absent':
                    present_today.append(entry[0])

# 3. Find the absentees and log them
markAbsentees(all_registered_students, present_today)

# Release the camera and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
print("Application closed.")