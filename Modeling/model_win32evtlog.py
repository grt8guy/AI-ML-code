#
# This script is part of the AI Models data source module.
#
# SCOPE: # This script is designed to gather error logs from Windows Event Logs and save them in CSV format
#
# CREATEDBY: John W. Braunsdorf
#
# CREATEDDATE: 2025-17-07
# 
# gathered windows event logs error data
# This script fetches error events from Windows Event Logs and saves them to CSV files.
# 

# Data gathering event error logs
import win32evtlog
import csv
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import socket

def hostname():
    return socket.gethostname()

device = hostname()

log_types = ['Application', 'Setup', 'System']
output_dir = fr'c:\temp\{device}'
os.makedirs(output_dir, exist_ok=True)

def fetch_events(log_type, max_events=1000):
    server = 'localhost'
    try:
        handle = win32evtlog.OpenEventLog(server, log_type)
    except Exception as e:
        print(f"Error accessing {log_type} log: {e}")
        return []
    flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
    events = []
    total = win32evtlog.GetNumberOfEventLogRecords(handle)
    read = 0
    ERROR_EVENT_TYPE = 1  # Error events have EventType == 1
    while read < total and len(events) < max_events:
        try:
            records = win32evtlog.ReadEventLog(handle, flags, 0)
        except Exception as e:
            print(f"Error reading events from {log_type}: {e}")
            break
        if not records:
            break
        for event in records:
            if event.EventType == ERROR_EVENT_TYPE:
                try:
                    message = win32evtlog.FormatMessage(event) if hasattr(win32evtlog, 'FormatMessage') else ''
                except Exception as e:
                    message = f"Error formatting message: {e}"
                try:
                    events.append({
                        'TimeGenerated': event.TimeGenerated.Format(),
                        'SourceName': event.SourceName,
                        'EventID': event.EventID & 0xFFFF,
                        'EventType': event.EventType,
                        'EventCategory': event.EventCategory,
                        'ComputerName': event.ComputerName,
                        'Message': message,
                    })
                except Exception as e:
                    print(f"Error processing event: {e}")
                if len(events) >= max_events:
                    break
        read += len(records)
    win32evtlog.CloseEventLog(handle)
    return events

for log_type in log_types:
    events = fetch_events(log_type)
    output_file = os.path.join(output_dir, f'{log_type.lower()}_log.csv')
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        if events:
            writer = csv.DictWriter(f, fieldnames=events[0].keys())
            writer.writeheader()
            writer.writerows(events)
        else:
            writer = csv.DictWriter(f, fieldnames=['TimeGenerated', 'SourceName', 'EventID', 'EventType', 'EventCategory', 'ComputerName', 'Message'])
            writer.writeheader()

# Collect and prepare data for model training
all_events = []
for log_type in log_types:
    file_path = os.path.join(output_dir, f'{log_type.lower()}_log.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['LogType'] = log_type
        all_events.append(df)

if all_events:
    data = pd.concat(all_events, ignore_index=True)
    data.fillna('', inplace=True)
    # Feature engineering: convert categorical columns to numeric
    data['SourceName'] = data['SourceName'].astype('category').cat.codes
    data['ComputerName'] = data['ComputerName'].astype('category').cat.codes
    data['LogType'] = data['LogType'].astype('category').cat.codes

    # Modeling for EventID and EventType
    # Predict EventType using EventID and other features
    X = data[['SourceName', 'EventID', 'EventCategory', 'ComputerName', 'LogType']]
    y_eventtype = data['EventType']
    X_train_et, X_test_et, y_train_et, y_test_et = train_test_split(X, y_eventtype, test_size=0.2, random_state=42)
    clf_eventtype = RandomForestClassifier(random_state=42)
    clf_eventtype.fit(X_train_et, y_train_et)
    y_pred_et = clf_eventtype.predict(X_test_et)
    print("EventType Classification Report:")
    print(classification_report(y_test_et, y_pred_et))
    print("EventType Confusion Matrix:")
    print(confusion_matrix(y_test_et, y_pred_et))
    joblib.dump(clf_eventtype, os.path.join(output_dir, 'eventtype_model.pkl'))

    # Predict EventID using other features (excluding EventID itself)
    X_eventid = data[['SourceName', 'EventType', 'EventCategory', 'ComputerName', 'LogType']]
    y_eventid = data['EventID']
    X_train_eid, X_test_eid, y_train_eid, y_test_eid = train_test_split(X_eventid, y_eventid, test_size=0.2, random_state=42)
    clf_eventid = RandomForestClassifier(random_state=42)
    clf_eventid.fit(X_train_eid, y_train_eid)
    y_pred_eid = clf_eventid.predict(X_test_eid)
    print("EventID Classification Report:")
    print(classification_report(y_test_eid, y_pred_eid))
    joblib.dump(clf_eventid, os.path.join(output_dir, 'eventid_model.pkl'))
else:
    print("No event data found for model training.")

# Example: Load trained model and predict on new data
def predict_eventtype(new_data):
    model_path = os.path.join(output_dir, 'eventtype_model.pkl')
    if not os.path.exists(model_path):
        print("EventType model not found.")
        return None
    clf = joblib.load(model_path)
    # Assume new_data is a DataFrame with the same feature columns as X
    preds = clf.predict(new_data)
    return preds

def predict_eventid(new_data):
    model_path = os.path.join(output_dir, 'eventid_model.pkl')
    if not os.path.exists(model_path):
        print("EventID model not found.")
        return None
    clf = joblib.load(model_path)
    # Assume new_data is a DataFrame with the same feature columns as X_eventid
    preds = clf.predict(new_data)
    return preds

# Evaluate model performance on the test sets
if all_events:
    print("Evaluating EventType model:")
    print("Accuracy:", clf_eventtype.score(X_test_et, y_test_et))
    print("Classification Report:\n", classification_report(y_test_et, y_pred_et))
    print("Confusion Matrix:\n", confusion_matrix(y_test_et, y_pred_et))

    print("\nEvaluating EventID model:")
    print("Accuracy:", clf_eventid.score(X_test_eid, y_test_eid))
    print("Classification Report:\n", classification_report(y_test_eid, y_pred_eid))

    # Hyperparameter tuning for RandomForestClassifier using GridSearchCV

    def optimize_model(X, y):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=1, scoring='accuracy')
        grid_search.fit(X, y)
        print("Best parameters found:", grid_search.best_params_)
        print("Best cross-validation score:", grid_search.best_score_)
        return grid_search.best_estimator_

    # Optimize EventType model
    best_eventtype_model = optimize_model(X_train_et, y_train_et)
    joblib.dump(best_eventtype_model, os.path.join(output_dir, 'eventtype_model_optimized.pkl'))

    # Optimize EventID model
    best_eventid_model = optimize_model(X_train_eid, y_train_eid)
    joblib.dump(best_eventid_model, os.path.join(output_dir, 'eventid_model_optimized.pkl'))
else:
    print("No models to evaluate.")

# Deploy and save the optimized models
def deploy_models():
    eventtype_model_path = os.path.join(output_dir, 'eventtype_model_optimized.pkl')
    eventid_model_path = os.path.join(output_dir, 'eventid_model_optimized.pkl')
    if os.path.exists(eventtype_model_path) and os.path.exists(eventid_model_path):
        print(f"Optimized EventType model saved at: {eventtype_model_path}")
        print(f"Optimized EventID model saved at: {eventid_model_path}")
    else:
        print("Optimized models not found. Please run optimization first.")

deploy_models()

        