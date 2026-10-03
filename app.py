"""
╔══════════════════════════════════════════════════════╗
║        MEHENDI PLOTTER — MAIN BACKEND                ║
║        File: app.py                                  ║
╚══════════════════════════════════════════════════════╝
"""
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import os
import time
import threading
import serial
import traceback
import math
from gcode_generator import generate_gcode

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════
SERIAL_PORT  = 'COM3'
BAUD_RATE    = 115200
CAMERA_INDEX = None   # auto-detected at startup
# ══════════════════════════════════════════════════════

camera      = None
camera_lock = threading.Lock()


# ── Auto-detect external USB webcam ─────────────────────────────────
def find_best_camera():
    """
    Scans indices 0-4. Returns the highest index that opens successfully —
    external USB webcams are almost always at a higher index than the
    built-in laptop camera (which is always index 0).
    Falls back to 0 if nothing else is found.
    """
    print("🔍 Scanning for cameras...")
    found = []
    for idx in range(5):
        # Use DSHOW backend explicitly for reliable Windows USB cam detection
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                found.append(idx)
                print(f"   ✅ Camera found at index {idx}")
            cap.release()
        else:
            print(f"   ❌ No camera at index {idx}")

    if not found:
        print("   ⚠️  No cameras found at all — defaulting to index 0")
        return 0

    # Prefer the LAST found index = the external USB webcam
    best = found[-1]
    if best == 0 and len(found) == 1:
        print("   ℹ️  Only built-in camera found. Plug in the Zebronics webcam and restart.")
    else:
        print(f"   🌸 Using external webcam at index {best}")
    return best
plotter_ser = None

# ── Scale shared state ──
current_scale  = 1.0
current_design = 'flower'
scale_lock     = threading.Lock()

# ── Overlay position state ──
lock_cx           = 0
lock_cy           = 0
lock_size         = 0   
lost_frames       = 0
LOST_RESET_FRAMES = 45   

OVERLAY_BASE_PX = 200    

# ── Plotter state ──
plotter_state = {
    'status':   'idle',
    'lines':    [],
    'current':  0,
    'total':    0,
    'progress': 0,
    'msg':      'Waiting…',
}
plotter_lock = threading.Lock()
resume_event = threading.Event()
resume_event.set()   

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
_design_cache = {}   


# ══════════════════════════════════════════════════════
#  CAMERA
# ══════════════════════════════════════════════════════
def get_camera():
    global camera, CAMERA_INDEX
    if camera is None or not camera.isOpened():
        if CAMERA_INDEX is None:
            CAMERA_INDEX = find_best_camera()
        # CAP_DSHOW backend avoids the "backend can't be used by index" warning
        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not camera.isOpened():
            print(f"⚠️  Camera index {CAMERA_INDEX} failed. Trying fallback index 0...")
            CAMERA_INDEX = 0
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera
# ══════════════════════════════════════════════════════
#  HAND DETECTION
# ══════════════════════════════════════════════════════
def detect_hand(frame):
    fh, fw = frame.shape[:2]
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0,   133,  77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask  = cv2.inRange(ycrcb, lower, upper)

    # 1. Block face region (top 35%)
    mask[0:int(fh * 0.35), :] = 0   

    # 2. FOCUS ZONE: Block out the sides and bottom to ignore the wallpaper.
    # We only look for skin inside a central box.
    margin_x = int(fw * 0.25)  # Cut 25% off left and right sides
    margin_y = int(fh * 0.15)  # Cut 15% off the bottom
    focus_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.rectangle(focus_mask, (margin_x, int(fh * 0.35)), (fw - margin_x, fh - margin_y), 255, -1)
    
    # Apply the focus zone so the background is completely blacked out for the detector
    mask = cv2.bitwise_and(mask, focus_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.GaussianBlur(mask, (3, 3), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0, 0, 0, frame.copy()

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 8000:
        return False, 0, 0, 0, frame.copy()

    cmask = np.zeros_like(mask)
    cv2.drawContours(cmask, [largest], -1, 255, thickness=cv2.FILLED)
    dist  = cv2.distanceTransform(cmask, cv2.DIST_L2, 5)
    _, palm_radius_f, _, max_loc = cv2.minMaxLoc(dist)

    return True, int(max_loc[0]), int(max_loc[1]), int(palm_radius_f), frame.copy()


def calc_scale(palm_radius_px):
    palm_diam = palm_radius_px * 2
    sc = palm_diam / OVERLAY_BASE_PX
    return round(max(0.3, min(2.0, sc)), 2)


# ══════════════════════════════════════════════════════
#  DESIGN IMAGE CACHE
# ══════════════════════════════════════════════════════
def _load_design(design_name):
    if design_name in _design_cache:
        return _design_cache[design_name]
    img_path = os.path.join(BASE_DIR, 'static', 'designs', f"{design_name}.png")
    if not os.path.exists(img_path):
        _design_cache[design_name] = None
        return None
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        _design_cache[design_name] = None
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    _, am = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    img[:, :, 3] = cv2.bitwise_and(img[:, :, 3], am)
    _design_cache[design_name] = img
    return img


# ══════════════════════════════════════════════════════
#  DESIGN OVERLAY
# ══════════════════════════════════════════════════════
def draw_design_overlay(frame, cx, cy, size_px, design_name):
    try:
        design_img = _load_design(design_name)
        if design_img is None:
            return

        h_src, w_src = design_img.shape[:2]
        new_w = max(1, int(size_px))
        new_h = max(1, int(h_src / w_src * size_px))
        resized = cv2.resize(design_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        x1 = cx - new_w // 2;  y1 = cy - new_h // 2
        x2 = x1 + new_w;       y2 = y1 + new_h
        fh, fw = frame.shape[:2]

        sx1 = max(0, -x1);   sy1 = max(0, -y1)
        sx2 = new_w - max(0, x2 - fw)
        sy2 = new_h - max(0, y2 - fh)
        dx1 = max(0, x1);    dy1 = max(0, y1)
        dx2 = min(fw, x2);   dy2 = min(fh, y2)

        if sx1 >= sx2 or sy1 >= sy2:
            return

        a   = resized[sy1:sy2, sx1:sx2, 3:4].astype(np.float32) / 255.0
        ai  = 1.0 - a
        roi = frame[dy1:dy2, dx1:dx2].astype(np.float32)
        src = resized[sy1:sy2, sx1:sx2, :3].astype(np.float32)
        frame[dy1:dy2, dx1:dx2] = np.clip(a * src + ai * roi, 0, 255).astype(np.uint8)

    except Exception as e:
        print(f"Overlay Error: {e}")


# ══════════════════════════════════════════════════════
#  VIDEO STREAM
# ══════════════════════════════════════════════════════
def generate_frames(design_name):
    global lock_cx, lock_cy, lock_size, lost_frames
    cam = get_camera()
    _load_design(design_name)   

    while True:
        with camera_lock:
            success, frame = cam.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        detected, palm_cx, palm_cy, palm_radius, annotated = detect_hand(frame)
        
        # Draw the Focus Zone box on screen so the user knows where to put their hand
        fh, fw = annotated.shape[:2]
        margin_x = int(fw * 0.25)
        margin_y = int(fh * 0.15)
        cv2.rectangle(annotated, (margin_x, int(fh * 0.35)), (fw - margin_x, fh - margin_y), (255, 255, 255), 1)
        cv2.putText(annotated, "Place Hand Inside Box", (margin_x + 5, int(fh * 0.35) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        with scale_lock:
            sc  = current_scale
            dsn = current_design

        if detected:
            lost_frames = 0
            palm_diam   = palm_radius * 2

            if lock_cx == 0 and lock_cy == 0:
                lock_cx   = palm_cx
                lock_cy   = palm_cy
                lock_size = palm_diam
            else:
                drift = math.hypot(palm_cx - lock_cx, palm_cy - lock_cy)
                
                # STRICT SNAP LOGIC: 
                # If drift < 20 pixels, do absolutely nothing. The design is completely frozen.
                # If drift > 20 pixels, instantly jump to the new spot. Zero lag.
                if drift > 20: 
                    lock_cx = palm_cx
                    lock_cy = palm_cy
                
                # Snap scale only if it changes significantly
                if abs(palm_diam - lock_size) > 15:
                    lock_size = palm_diam

            overlay_px = int(OVERLAY_BASE_PX * sc)
            draw_design_overlay(annotated, lock_cx, lock_cy, overlay_px, dsn)
            cv2.putText(annotated,
                        f"Scale:{sc:.2f}x  {int(80*sc)}mm  palm:{palm_diam}px",
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 220, 120), 1)
        else:
            lost_frames += 1
            if lost_frames >= LOST_RESET_FRAMES:
                lock_cx = 0;  lock_cy = 0;  lock_size = 0
            elif lock_cx != 0:
                overlay_px = int(OVERLAY_BASE_PX * sc)
                draw_design_overlay(annotated, lock_cx, lock_cy, overlay_px, dsn)
                cv2.putText(annotated, "Hand not detected — overlay frozen",
                            (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 140, 220), 1)

        ret, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


# ══════════════════════════════════════════════════════
#  SERIAL
# ══════════════════════════════════════════════════════
def connect_plotter():
    global plotter_ser
    try:
        plotter_ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)
        plotter_ser.flushInput()
        print(f"✅  Plotter connected on {SERIAL_PORT}")
        return True
    except Exception:
        print("⚠️   Plotter not connected — simulation mode.")
        plotter_ser = None
        return False


def _send_line(line):
    if plotter_ser is None or not plotter_ser.is_open:
        time.sleep(0.2)  # Slowed down for simulation preview
        return
    plotter_ser.write((line + '\n').encode())
    while True:
        resp = plotter_ser.readline().decode('utf-8', errors='ignore').strip()
        if resp == 'ok':
            break
        elif resp.startswith('error'):
            print(f"⚠️ GRBL: {resp}  ← {line}")
            break


def _prep_lines(gcode):
    out = []
    for raw in gcode.split('\n'):
        l = raw.strip()
        if ';' in l:
            l = l.split(';')[0].strip()
        if l:
            out.append(l)
    return out


# ══════════════════════════════════════════════════════
#  PLOTTER WORKER THREAD
# ══════════════════════════════════════════════════════
def _plotter_worker():
    with plotter_lock:
        lines  = list(plotter_state['lines'])
        start  = plotter_state['current']
        total  = plotter_state['total']
        plotter_state['status'] = 'running'

    if plotter_ser and plotter_ser.is_open:
        plotter_ser.write(b'\r\n\r\n')
        time.sleep(2)
        plotter_ser.reset_input_buffer()
        plotter_ser.write(b'$X\n')   
        time.sleep(0.5)              

    for idx in range(start, len(lines)):
        with plotter_lock:
            if plotter_state['status'] == 'idle':
                return

        resume_event.wait()   

        with plotter_lock:
            if plotter_state['status'] == 'idle':
                return

        _send_line(lines[idx])

        with plotter_lock:
            plotter_state['current']  = idx + 1
            plotter_state['progress'] = min(int((idx + 1) / max(total, 1) * 100), 100)
            plotter_state['msg']      = f"Line {idx+1}/{total}: {lines[idx][:40]}"

    with plotter_lock:
        plotter_state['status']   = 'done'
        plotter_state['progress'] = 100
        plotter_state['msg']      = '✅ Print complete!'


# ══════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/preview/<design_name>')
def preview(design_name):
    return render_template('preview.html', design=design_name)


@app.route('/video_feed')
def video_feed():
    global current_design
    design_name = request.args.get('design', 'flower')
    with scale_lock:
        current_design = design_name
    return Response(generate_frames(design_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_scale', methods=['POST'])
def set_scale():
    global current_scale, current_design
    data = request.get_json(silent=True) or {}
    with scale_lock:
        current_scale  = float(data.get('scale',  current_scale))
        current_design = data.get('design', current_design)
    return jsonify({'ok': True, 'scale': current_scale})


@app.route('/detect_hand', methods=['POST'])
def detect_hand_api():
    cam = get_camera()
    with camera_lock:
        ok, frame = cam.read()
    if not ok:
        return jsonify({'detected': False, 'scale': 1.0})
    frame = cv2.flip(frame, 1)
    detected, palm_cx, palm_cy, palm_radius, _ = detect_hand(frame)
    if detected:
        sc = calc_scale(palm_radius)
        return jsonify({'detected': True, 'scale': sc,
                        'hand_width_px': palm_radius * 2,
                        'size_mm': round(80 * sc, 1)})
    return jsonify({'detected': False, 'scale': 1.0, 'hand_width_px': 0})


@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    global lock_cx, lock_cy, lock_size
    try:
        data   = request.get_json(silent=True) or {}
        scale  = float(data.get('scale', 1.0))
        design = data.get('design', 'flower')

        cam = get_camera()
        with camera_lock:
            ok, frame = cam.read()
        if not ok:
            return jsonify({'success': False, 'error': 'Camera read failed'})

        frame = cv2.flip(frame, 1)
        detected, palm_cx, palm_cy, palm_radius, annotated = detect_hand(frame)

        # Force capture to use the stable, frozen coordinates first
        if lock_cx != 0 and lock_cy != 0:
            cx, cy  = lock_cx, lock_cy  
        elif detected:
            cx, cy  = palm_cx, palm_cy
        else:
            fh, fw  = frame.shape[:2]
            cx, cy  = fw // 2, fh // 2

        size_px = int(OVERLAY_BASE_PX * scale)
        draw_design_overlay(annotated, cx, cy, size_px, design)

        os.makedirs(os.path.join(BASE_DIR, 'static'), exist_ok=True)
        cv2.imwrite(os.path.join(BASE_DIR, 'static', 'captured_frame.jpg'), annotated)

        gcode = generate_gcode(design, scale)
        with open(os.path.join(BASE_DIR, 'static', 'output.gcode'), 'w', encoding='utf-8') as f:
            f.write(gcode)

        real_lines = [l for l in gcode.split('\n') if l.strip() and not l.startswith(';')]
        return jsonify({
            'success':        True,
            'frame':          '/static/captured_frame.jpg',
            'gcode_lines':    len(real_lines),
            'gcode_preview':  '\n'.join(gcode.split('\n')[:25]),
            'scale':          scale,
            'design_size_mm': round(80 * scale, 1)
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


# ── START ──────────────────────────────────────────────────────────
@app.route('/start_plotter', methods=['POST'])
def start_plotter():
    try:
        with open(os.path.join(BASE_DIR, 'static', 'output.gcode'), 'r', encoding='utf-8') as f:
            gcode = f.read()
        lines = _prep_lines(gcode)
        with plotter_lock:
            plotter_state.update({
                'status':   'running',
                'lines':    lines,
                'current':  0,
                'total':    len(lines),
                'progress': 0,
                'msg':      'Starting…',
            })
        resume_event.set()   
        threading.Thread(target=_plotter_worker, daemon=True).start()
        return jsonify({'success': True, 'message': 'Plotter started!'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── STOP ───────────────────────────────────────────────────────────
@app.route('/stop_plotter', methods=['POST'])
def stop_plotter():
    resume_event.clear()   
    with plotter_lock:
        cur = plotter_state['current']
        tot = plotter_state['total']
        plotter_state['status'] = 'stopped'
        plotter_state['msg']    = f"Paused at line {cur}/{tot}"
    if plotter_ser and plotter_ser.is_open:
        plotter_ser.write(b'!\n')   
    return jsonify({'success': True, 'message': f'Paused at line {cur}/{tot}.'})


# ── RESUME ─────────────────────────────────────────────────────────
@app.route('/resume_plotter', methods=['POST'])
def resume_plotter():
    with plotter_lock:
        status  = plotter_state['status']
        current = plotter_state['current']
        total   = plotter_state['total']

    if status == 'done':
        return jsonify({'success': False, 'error': 'Print already finished.'})
    if status == 'idle':
        return jsonify({'success': False, 'error': 'Nothing to resume.'})
    if status == 'running':
        return jsonify({'success': True, 'message': 'Already running.'})

    with plotter_lock:
        plotter_state['status'] = 'running'
        plotter_state['msg']    = f'Resuming from line {current}/{total}…'

    if plotter_ser and plotter_ser.is_open:
        plotter_ser.write(b'~\n')   

    resume_event.set()
    threading.Thread(target=_plotter_worker, daemon=True).start()
    return jsonify({'success': True, 'message': f'Resumed from line {current}.'})


# ── RESET ──────────────────────────────────────────────────────────
@app.route('/reset_plotter', methods=['POST'])
def reset_plotter():
    global lock_cx, lock_cy, lock_size, lost_frames
    resume_event.clear()
    with plotter_lock:
        plotter_state.update({
            'status':   'idle',
            'lines':    [],
            'current':  0,
            'total':    0,
            'progress': 0,
            'msg':      'Waiting…',
        })

    if plotter_ser and plotter_ser.is_open:
        plotter_ser.write(b'!\n')
        time.sleep(0.15)
        plotter_ser.write(b'~\n')
        time.sleep(0.15)
        plotter_ser.write(b'M03 S90\n')   
        time.sleep(0.15)
        plotter_ser.write(b'G1 F1500 X0 Y0\n')

    lock_cx = 0;  lock_cy = 0;  lock_size = 0;  lost_frames = 0
    resume_event.set()   
    return jsonify({'success': True, 'message': 'Reset done. Pen down. Ready.'})


# ── STATUS ─────────────────────────────────────────────────────────
@app.route('/plotter_status')
def plotter_status():
    with plotter_lock:
        s = dict(plotter_state)
    return jsonify({
        'status':   s['status'],
        'progress': s['progress'],
        'current':  s['current'],
        'total':    s['total'],
        'msg':      s['msg'],
        'running':  s['status'] == 'running',
        'stopped':  s['status'] == 'stopped',
        'finished': s['status'] == 'done',
        'idle':     s['status'] == 'idle',
    })


@app.route('/home_plotter', methods=['POST'])
def home_plotter():
    if plotter_ser and plotter_ser.is_open:
        plotter_ser.write(b'$H\n')
    return jsonify({'success': True})


if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'static', 'designs'), exist_ok=True)
    CAMERA_INDEX = find_best_camera()   # detect external webcam before anything else
    connect_plotter()
    print("\n🌸 Mehendi Plotter running at http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)