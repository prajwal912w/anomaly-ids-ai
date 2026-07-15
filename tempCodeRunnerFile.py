import tkinter as tk
from tkinter import scrolledtext, filedialog
from scapy.all import sniff
import numpy as np
import joblib
import datetime
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import winsound

# Load model
model = joblib.load("ids_model.pkl")
scaler = joblib.load("scaler.pkl")
EXPECTED_FEATURES = scaler.n_features_in_

# Stats
normal_count = 0
attack_count = 0
total_packets = 0

history_normal = []
history_attack = []

running = False

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
# DETECTION
# -----------------------------
def detect(pkt):
    global normal_count, attack_count, total_packets

    if not running:
        return

    total_packets += 1
    features = extract_features(pkt)

    if features[0] < 60 or features == [0,0,0,0]:
        return

    padded = features + [0] * (EXPECTED_FEATURES - len(features))
    padded = padded[:EXPECTED_FEATURES]

    data = np.array(padded).reshape(1, -1)
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)
    score = model.decision_function(data_scaled)[0]

    time = datetime.datetime.now().strftime("%H:%M:%S")

    # Severity
    if features[0] > 1500:
        severity = "HIGH"
    elif features[0] > 800:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    if features[3] not in [80, 443, 53]:
        severity = "MEDIUM"

    if prediction[0] == -1:
        attack_count += 1
        msg = f"{time} ALERT [{severity}] Score:{round(score,2)} | {features}\n"
        log_box.insert(tk.END, msg, "alert")

        winsound.Beep(1000, 200)  # sound alert

        with open("attack_logs.txt", "a") as f:
            f.write(msg)
    else:
        normal_count += 1
        msg = f"{time} NORMAL | {features}\n"
        log_box.insert(tk.END, msg, "normal")

    log_box.yview(tk.END)

    update_stats()
    update_graph()

# -----------------------------
# GRAPH
# -----------------------------
def update_graph():
    ax.clear()

    ax.plot(history_normal, color="green", linewidth=2, label="Normal")
    ax.plot(history_attack, color="red", linewidth=2, label="Attack")

    ax.set_facecolor("#f1f5f9")
    ax.set_title("Live Traffic Trend")
    ax.legend()

    canvas.draw()

def update_stats():
    total_label.config(text=f"Total: {total_packets}")
    normal_label.config(text=f"Normal: {normal_count}")
    attack_label.config(text=f"Attacks: {attack_count}")

    history_normal.append(normal_count)
    history_attack.append(attack_count)

# -----------------------------
# CONTROL FUNCTIONS
# -----------------------------
def start_ids():
    global running
    running = True
    status_label.config(text="Status: Running", fg="green")
    sniff(prn=detect, store=0)

def stop_ids():
    global running
    running = False
    status_label.config(text="Status: Stopped", fg="red")

def run_ids():
    threading.Thread(target=start_ids, daemon=True).start()

def clear_logs():
    log_box.delete(1.0, tk.END)

def export_logs():
    file = filedialog.asksaveasfilename(defaultextension=".txt")
    if file:
        with open(file, "w") as f:
            f.write(log_box.get(1.0, tk.END))

# -----------------------------
# GUI
# -----------------------------
root = tk.Tk()
root.title("IDS Dashboard")
root.geometry("1000x750")
root.configure(bg="#e2e8f0")

# Title
title = tk.Label(root, text="REAL-TIME INTRUSION DETECTION SYSTEM",
                 font=("Arial", 16, "bold"),
                 bg="#e2e8f0")
title.pack(pady=10)

# Status
status_label = tk.Label(root, text="Status: Stopped", fg="red", bg="#e2e8f0")
status_label.pack()

# Stats
frame = tk.Frame(root, bg="#e2e8f0")
frame.pack()

total_label = tk.Label(frame, text="Total: 0", bg="#e2e8f0")
total_label.grid(row=0, column=0, padx=20)

normal_label = tk.Label(frame, text="Normal: 0", fg="green", bg="#e2e8f0")
normal_label.grid(row=0, column=1, padx=20)

attack_label = tk.Label(frame, text="Attacks: 0", fg="red", bg="#e2e8f0")
attack_label.grid(row=0, column=2, padx=20)

# Logs
log_box = scrolledtext.ScrolledText(root, width=110, height=15,
                                    bg="white", fg="black")
log_box.pack(pady=10)

log_box.tag_config("alert", foreground="red")
log_box.tag_config("normal", foreground="green")

# Buttons
btn_frame = tk.Frame(root, bg="#e2e8f0")
btn_frame.pack()

tk.Button(btn_frame, text="Start", command=run_ids).grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="Stop", command=stop_ids).grid(row=0, column=1, padx=10)
tk.Button(btn_frame, text="Clear Logs", command=clear_logs).grid(row=0, column=2, padx=10)
tk.Button(btn_frame, text="Export Logs", command=export_logs).grid(row=0, column=3, padx=10)

# Graph
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

root.mainloop()