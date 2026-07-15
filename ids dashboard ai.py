"""
AI-Enhanced Anomaly-Based Intrusion Detection System
Adds LLM-powered threat intelligence and MITRE ATT&CK mapping via Anthropic API
"""

import tkinter as tk
from tkinter import scrolledtext, filedialog, ttk
import numpy as np
import joblib
import datetime
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import requests
import json
import queue
import os
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_API_KEY_HERE")
AI_KEY_VALID = GROQ_API_KEY != "YOUR_API_KEY_HERE" and len(GROQ_API_KEY) > 10
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.1-8b-instant"

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATED PACKET DATA (used when Scapy / live capture not available)
# ─────────────────────────────────────────────────────────────────────────────
ATTACK_PROFILES = [
    {"name": "Port Scan",        "size": 64,   "proto": 1, "sport": 12345, "dport": 22},
    {"name": "SYN Flood",        "size": 60,   "proto": 1, "sport": 54321, "dport": 80},
    {"name": "DNS Amplification","size": 512,  "proto": 2, "sport": 53,   "dport": 53},
    {"name": "SQL Injection",    "size": 1200, "proto": 1, "sport": 44321, "dport": 3306},
    {"name": "Brute Force SSH",  "size": 200,  "proto": 1, "sport": 55000, "dport": 22},
    {"name": "Data Exfil",       "size": 8000, "proto": 1, "sport": 9999,  "dport": 443},
]

MITRE_MAP = {
    "Port Scan":         ("TA0007 – Discovery",           "T1046 – Network Service Scanning"),
    "SYN Flood":         ("TA0040 – Impact",              "T1498 – Network Denial of Service"),
    "DNS Amplification": ("TA0040 – Impact",              "T1498.002 – Reflection Amplification"),
    "SQL Injection":     ("TA0001 – Initial Access",      "T1190 – Exploit Public-Facing Application"),
    "Brute Force SSH":   ("TA0006 – Credential Access",   "T1110 – Brute Force"),
    "Data Exfil":        ("TA0010 – Exfiltration",        "T1041 – Exfiltration Over C2 Channel"),
}

# ─────────────────────────────────────────────────────────────────────────────
# AI ANALYSIS MODULE
# ─────────────────────────────────────────────────────────────────────────────
ai_queue = queue.Queue()   # worker → UI thread

def query_ai(features: dict) -> str:
    """Call Groq API to get threat intelligence for detected anomaly."""
    prompt = f"""You are a senior SOC analyst. Analyze this network anomaly:

Packet features: {json.dumps(features, indent=2)}

Provide a concise threat intelligence report covering:
1. Attack type classification
2. MITRE ATT&CK tactic and technique
3. Severity assessment (Critical/High/Medium/Low)
4. Recommended immediate remediation steps (max 3 bullet points)

Keep it brief – max 200 words."""

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
            },
            timeout=20,
        )
        data = resp.json()
        if "error" in data:
            return f"[API Error: {data['error'].get('message', data['error'])}]"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[AI Analysis unavailable: {e}]"


def ai_worker():
    """Background thread that drains the analysis queue."""
    while True:
        task = ai_queue.get()
        if task is None:
            break
        features, callback = task
        result = query_ai(features)
        root.after(0, callback, result)
        ai_queue.task_done()


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION ENGINE  (uses pre-trained Isolation Forest via joblib)
# ─────────────────────────────────────────────────────────────────────────────
try:
    model_ml    = joblib.load("ids_model.pkl")
    scaler_ml   = joblib.load("scaler.pkl")
    EXPECTED_F  = scaler_ml.n_features_in_
    USE_REAL_ML = True
    ML_STATUS   = "✅ ML model loaded (ids_model.pkl + scaler.pkl)"
except Exception as e:
    USE_REAL_ML = False
    ML_STATUS   = f"⚠ ML models not found – using heuristic fallback ({e})"

normal_count = 0
attack_count = 0
total_packets = 0
history_normal: list[int] = []
history_attack:  list[int] = []
running = False
ai_pending = 0   # track how many AI requests are in-flight


def classify_packet(size: int, proto: int, sport: int, dport: int) -> int:
    """Returns -1 (attack) or 1 (normal)."""
    if USE_REAL_ML:
        feats = [size, proto, sport, dport]
        padded = (feats + [0] * (EXPECTED_F - len(feats)))[:EXPECTED_F]
        scaled = scaler_ml.transform(np.array(padded).reshape(1, -1))
        return int(model_ml.predict(scaled)[0])
    # Heuristic fallback
    if dport in (22, 3306, 5432) and sport > 50000:
        return -1
    if size > 5000 and dport in (443, 80):
        return -1
    if dport == 53 and size > 300:
        return -1
    return 1


def severity_from_features(size: int, proto: int, dport: int) -> str:
    if size > 5000 or dport in (22, 3306, 5432):
        return "CRITICAL"
    if size > 1500 or dport == 53:
        return "HIGH"
    if size > 800:
        return "MEDIUM"
    return "LOW"


def process_packet(size: int, proto: int, sport: int, dport: int,
                   attack_name: str = "Unknown") -> None:
    global normal_count, attack_count, total_packets, ai_pending

    total_packets += 1
    prediction = classify_packet(size, proto, sport, dport)
    ts = datetime.datetime.now().strftime("%H:%M:%S")

    if prediction == -1:
        attack_count += 1
        sev = severity_from_features(size, proto, dport)
        mitre = MITRE_MAP.get(attack_name, ("TA????", "T????"))

        msg = (f"{ts}  ⚠  ATTACK [{sev}] | {attack_name} | "
               f"MITRE: {mitre[1]} | "
               f"size={size} proto={'TCP' if proto==1 else 'UDP'} "
               f"sport={sport} dport={dport}\n")
        log_box.insert(tk.END, msg, "alert")

        # Queue AI analysis for HIGH/CRITICAL alerts
        if sev in ("HIGH", "CRITICAL") and ai_pending < 3:
            ai_pending += 1
            features_dict = {
                "timestamp": ts,
                "attack_name": attack_name,
                "size": size,
                "protocol": "TCP" if proto == 1 else "UDP",
                "src_port": sport,
                "dst_port": dport,
                "severity": sev,
                "mitre_tactic": mitre[0],
                "mitre_technique": mitre[1],
            }
            ai_queue.put((features_dict, on_ai_result))
            ai_status.config(text=f"🤖 AI analysing… ({ai_pending} pending)")
    else:
        normal_count += 1
        msg = f"{ts}  ✓  NORMAL | size={size} proto={'TCP' if proto==1 else 'UDP'} sport={sport} dport={dport}\n"
        log_box.insert(tk.END, msg, "normal")

    log_box.yview(tk.END)
    update_stats()
    update_graph()

    with open("attack_logs.txt", "a", encoding="utf-8") as f:
        f.write(msg)


def on_ai_result(text: str) -> None:
    """Called on the main thread when AI analysis returns."""
    global ai_pending
    ai_pending = max(0, ai_pending - 1)
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    ai_panel.config(state=tk.NORMAL)
    ai_panel.insert(tk.END, f"\n{'─'*60}\n[{ts}] AI Threat Analysis:\n{text}\n")
    ai_panel.see(tk.END)
    ai_panel.config(state=tk.DISABLED)
    ai_status.config(text=f"🤖 AI ready  ({ai_pending} pending)")


# ─────────────────────────────────────────────────────────────────────────────
# LIVE CAPTURE (tries Scapy; falls back to simulation)
# ─────────────────────────────────────────────────────────────────────────────
def scapy_sniff() -> None:
    try:
        from scapy.all import sniff as _sniff

        def _handle(pkt):
            if not running:
                return
            size  = len(pkt)
            proto = 1 if pkt.haslayer("TCP") else (2 if pkt.haslayer("UDP") else 0)
            sport = getattr(pkt, "sport", 0)
            dport = getattr(pkt, "dport", 0)
            if size < 60:
                return
            root.after(0, process_packet, size, proto, sport, dport, "Unknown")

        _sniff(prn=_handle, store=0, stop_filter=lambda _: not running)
    except Exception:
        simulate_traffic()


def simulate_traffic() -> None:
    """Generate synthetic normal + attack packets for demo purposes."""
    while running:
        if random.random() < 0.15:         # 15 % chance of attack packet
            profile = random.choice(ATTACK_PROFILES)
            root.after(0, process_packet,
                       profile["size"], profile["proto"],
                       profile["sport"], profile["dport"],
                       profile["name"])
        else:
            size  = random.randint(64, 1500)
            proto = random.choice([1, 2])
            sport = random.randint(1024, 65535)
            dport = random.choice([80, 443, 53, 8080, 22])
            root.after(0, process_packet, size, proto, sport, dport, "Normal")
        time.sleep(random.uniform(0.3, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────────────────────────────────────
def update_graph():
    ax.clear()
    ax.plot(history_normal[-60:], color="#22c55e", linewidth=2, label="Normal")
    ax.plot(history_attack[-60:],  color="#ef4444", linewidth=2, label="Attack")
    ax.set_facecolor("#f8fafc")
    ax.set_title("Live Traffic Trend (last 60 samples)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    canvas.draw()


def update_stats():
    total_label.config(text=f"Total: {total_packets}")
    normal_label.config(text=f"Normal: {normal_count}")
    attack_label.config(text=f"Attacks: {attack_count}")
    history_normal.append(normal_count)
    history_attack.append(attack_count)


# ─────────────────────────────────────────────────────────────────────────────
# CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
def run_ids():
    global running
    running = True
    status_label.config(text="Status: ● Running", fg="#16a34a")
    threading.Thread(target=scapy_sniff, daemon=True).start()


def stop_ids():
    global running
    running = False
    status_label.config(text="Status: ■ Stopped", fg="#dc2626")


def clear_logs():
    log_box.delete(1.0, tk.END)


def export_logs():
    path = filedialog.asksaveasfilename(defaultextension=".txt",
                                        filetypes=[("Text", "*.txt")])
    if path:
        with open(path, "w") as f:
            f.write(log_box.get(1.0, tk.END))


def ask_ai_manually():
    """User can type a custom query in the AI panel."""
    query_win = tk.Toplevel(root)
    query_win.title("Ask AI Analyst")
    query_win.geometry("500x200")
    tk.Label(query_win, text="Describe a threat or paste log lines:").pack(pady=4)
    entry = scrolledtext.ScrolledText(query_win, height=6)
    entry.pack(fill=tk.BOTH, expand=True, padx=8)

    def submit():
        text = entry.get("1.0", tk.END).strip()
        if text:
            ai_queue.put(({"user_query": text}, on_ai_result))
            ai_status.config(text="🤖 AI analysing…")
        query_win.destroy()

    tk.Button(query_win, text="Analyse", command=submit,
              bg="#2563eb", fg="white").pack(pady=6)


# ─────────────────────────────────────────────────────────────────────────────
# GUI LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("AI-Enhanced IDS Dashboard")
root.geometry("1200x820")
root.configure(bg="#f1f5f9")

# ── Title ──
tk.Label(root,
         text="🔐  AI-Enhanced Real-Time Intrusion Detection System",
         font=("Arial", 15, "bold"), bg="#1e3a5f", fg="white",
         pady=10).pack(fill=tk.X)

# ── Status row ──
status_row = tk.Frame(root, bg="#f1f5f9")
status_row.pack(fill=tk.X, padx=10, pady=4)

status_label = tk.Label(status_row, text="Status: ■ Stopped",
                         fg="#dc2626", bg="#f1f5f9", font=("Arial", 11, "bold"))
status_label.pack(side=tk.LEFT, padx=8)

ai_status = tk.Label(status_row,
                      text="🤖 AI ready" if AI_KEY_VALID else "🤖 AI – no API key",
                      fg="#2563eb" if AI_KEY_VALID else "#dc2626",
                      bg="#f1f5f9", font=("Arial", 10))
ai_status.pack(side=tk.LEFT, padx=20)

# ── Stats row ──
stats_row = tk.Frame(root, bg="#f1f5f9")
stats_row.pack(fill=tk.X, padx=10)

total_label  = tk.Label(stats_row, text="Total: 0",   bg="#f1f5f9", font=("Arial", 10))
normal_label = tk.Label(stats_row, text="Normal: 0",  bg="#f1f5f9", fg="#16a34a", font=("Arial", 10, "bold"))
attack_label = tk.Label(stats_row, text="Attacks: 0", bg="#f1f5f9", fg="#dc2626", font=("Arial", 10, "bold"))
for lbl in (total_label, normal_label, attack_label):
    lbl.pack(side=tk.LEFT, padx=20)

# ── Main paned area ──
paned = tk.PanedWindow(root, orient=tk.HORIZONTAL, bg="#f1f5f9", sashwidth=6)
paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

# Left: log + buttons
left_frame = tk.Frame(paned, bg="#f1f5f9")
paned.add(left_frame, minsize=580)

tk.Label(left_frame, text="Live Packet Log", font=("Arial", 10, "bold"),
         bg="#f1f5f9").pack(anchor=tk.W)

log_box = scrolledtext.ScrolledText(left_frame, width=80, height=20,
                                     bg="white", fg="#111827",
                                     font=("Courier", 9))
log_box.pack(fill=tk.BOTH, expand=True, pady=(0, 4))
log_box.tag_config("alert",  foreground="#dc2626", font=("Courier", 9, "bold"))
log_box.tag_config("normal", foreground="#16a34a")

# Graph
fig, ax = plt.subplots(figsize=(6, 2.5), dpi=90)
canvas = FigureCanvasTkAgg(fig, master=left_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH)

# Buttons
btn_frame = tk.Frame(left_frame, bg="#f1f5f9")
btn_frame.pack(pady=6)
btn_cfg = {"font": ("Arial", 9, "bold"), "relief": tk.FLAT, "padx": 10, "pady": 4}
tk.Button(btn_frame, text="▶ Start",       command=run_ids,       bg="#16a34a", fg="white", **btn_cfg).grid(row=0, column=0, padx=4)
tk.Button(btn_frame, text="■ Stop",        command=stop_ids,       bg="#dc2626", fg="white", **btn_cfg).grid(row=0, column=1, padx=4)
tk.Button(btn_frame, text="Clear Logs",    command=clear_logs,     bg="#6b7280", fg="white", **btn_cfg).grid(row=0, column=2, padx=4)
tk.Button(btn_frame, text="Export Logs",   command=export_logs,    bg="#374151", fg="white", **btn_cfg).grid(row=0, column=3, padx=4)
tk.Button(btn_frame, text="🤖 Ask AI",    command=ask_ai_manually, bg="#2563eb", fg="white", **btn_cfg).grid(row=0, column=4, padx=4)

# Right: AI panel
right_frame = tk.Frame(paned, bg="#f1f5f9")
paned.add(right_frame, minsize=340)

tk.Label(right_frame, text="🤖  AI Threat Intelligence (Groq / LLaMA3)",
         font=("Arial", 10, "bold"), bg="#f1f5f9").pack(anchor=tk.W)

ai_panel = scrolledtext.ScrolledText(right_frame, width=46, height=40,
                                      bg="#1e1e2e", fg="#cdd6f4",
                                      font=("Courier", 9),
                                      state=tk.DISABLED,
                                      insertbackground="white")
ai_panel.pack(fill=tk.BOTH, expand=True)

ai_panel.config(state=tk.NORMAL)

# Build startup status message with live checks
_key_line  = ("✅ GROQ_API_KEY detected – AI analysis enabled."
              if AI_KEY_VALID
              else "❌ GROQ_API_KEY not set!\n"
                   "   Set it in Environment Variables:\n"
                   "   Variable name : GROQ_API_KEY\n"
                   "   Variable value: your key from console.groq.com\n"
                   "   Then restart CMD and rerun.")

ai_panel.insert(tk.END,
    "AI Threat Intelligence Panel\n"
    "─────────────────────────────\n"
    "HIGH and CRITICAL anomalies are automatically\n"
    "analysed by LLaMA3 via Groq (free API).\n\n"
    "Use '🤖 Ask AI' to submit custom queries.\n\n"
    f"{_key_line}\n\n"
    f"{ML_STATUS}\n")
ai_panel.config(state=tk.DISABLED)

# ─────────────────────────────────────────────────────────────────────────────
# START BACKGROUND AI WORKER
# ─────────────────────────────────────────────────────────────────────────────
threading.Thread(target=ai_worker, daemon=True).start()

root.mainloop()
