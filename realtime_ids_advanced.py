from scapy.all import sniff
import numpy as np
import joblib
import datetime

# Load model
model = joblib.load("ids_model.pkl")
scaler = joblib.load("scaler.pkl")

EXPECTED_FEATURES = scaler.n_features_in_

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(pkt):
    try:
        size = len(pkt)

        if pkt.haslayer("TCP"):
            proto = 1
        elif pkt.haslayer("UDP"):
            proto = 2
        else:
            proto = 0

        sport = pkt.sport if hasattr(pkt, 'sport') else 0
        dport = pkt.dport if hasattr(pkt, 'dport') else 0

        return [size, proto, sport, dport]

    except:
        return [0, 0, 0, 0]

# -----------------------------
# DETECTION FUNCTION
# -----------------------------
def detect(pkt):
    try:
        features = extract_features(pkt)

        # Ignore very small packets
        if features[0] < 60:
            return

        # Match feature size
        padded = features + [0] * (EXPECTED_FEATURES - len(features))
        padded = padded[:EXPECTED_FEATURES]

        data = np.array(padded).reshape(1, -1)
        data_scaled = scaler.transform(data)

        prediction = model.predict(data_scaled)
        score = model.decision_function(data_scaled)[0]

        time = datetime.datetime.now().strftime("%H:%M:%S")

        # Severity logic
        if features[0] > 1500:
            severity = "HIGH"
        elif features[0] > 800:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Port logic
        if features[3] not in [80, 443, 53]:
            severity = "MEDIUM"

        # Final output
        if prediction[0] == -1:
            print(time + " ALERT [" + severity + "] Score:" + str(round(score,2)) + " | " + str(features))
        else:
            print(time + " Normal | " + str(features))

    except Exception as e:
        print("Error:", e)

# -----------------------------
# START IDS
# -----------------------------
print("Real-Time IDS Started...\n")

sniff(prn=detect, count=50)