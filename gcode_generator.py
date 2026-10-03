"""
╔══════════════════════════════════════════════════════╗
║        MEHENDI PLOTTER — GCODE GENERATOR             ║
║        File: gcode_generator.py                      ║
╚══════════════════════════════════════════════════════╝
"""
import os
import cv2
import numpy as np
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEED_DRAW   = 800    # mm/min drawing
FEED_TRAVEL = 1200   # mm/min travel between shapes (pen down)

# 0.65 = "if hole is >65% the size of its parent outline, it's a stroke wall."
HOLE_RATIO  = 0.65

# Minimum contour area in px² to keep (filters noise/dust specs)
MIN_AREA_PX = 15


def _dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _centroid(cnt):
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return float(cnt[0][0][0]), float(cnt[0][0][1])
    return M['m10'] / M['m00'], M['m01'] / M['m00']


def generate_gcode(design_name, scale=1.0):
    img_path = os.path.join(BASE_DIR, 'static', 'designs', f"{design_name}.png")
    if not os.path.exists(img_path):
        return f"; ERROR: Image not found at {img_path}"

    # ── 1. Load & flatten transparency ───────────────────────────
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return "; ERROR: Failed to load image."

    if len(img.shape) == 3 and img.shape[2] == 4:
        bg      = np.full_like(img[:, :, :3], 255)
        alpha   = img[:, :, 3:4].astype(np.float32) / 255.0
        img_rgb = (img[:, :, :3] * alpha + bg * (1 - alpha)).astype(np.uint8)
        gray    = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # ── 2. Threshold + small morphological close ─────────────────
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)

    h_px, w_px      = binary.shape
    target_width_mm = 80.0 * scale
    mm_per_px       = target_width_mm / w_px
    h_mm            = h_px * mm_per_px

    # ── 3. RETR_TREE — all contours with hierarchy ───────────────
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return "; ERROR: No contours found in design."

    hier  = hierarchy[0]                      
    areas = [cv2.contourArea(c) for c in contours]

    # ── 4. Filter: keep genuine design features, skip stroke walls ─
    kept_idx = []
    for i, (c, h) in enumerate(zip(contours, hier)):
        if areas[i] < MIN_AREA_PX:
            continue                           

        parent = h[3]
        if parent >= 0:
            p_area = areas[parent]
            if p_area > 0 and areas[i] / p_area >= HOLE_RATIO:
                continue                       

        kept_idx.append(i)

    if not kept_idx:
        return "; ERROR: No contours survived filtering."

    # ── 5. Deduplicate near-identical contours ───────────────────
    unique_idx = []
    used = set()
    for i in kept_idx:
        if i in used:
            continue
        cx_i, cy_i = _centroid(contours[i])
        unique_idx.append(i)
        for j in kept_idx:
            if j <= i or j in used:
                continue
            if areas[i] > 0:
                cx_j, cy_j = _centroid(contours[j])
                if (_dist((cx_i, cy_i), (cx_j, cy_j)) < 3.0 and
                        abs(areas[i] - areas[j]) / areas[i] < 0.06):
                    used.add(j)

    # ── 6. Simplify contours → mm closed loops ───────────────────
    loops = []
    for i in unique_idx:
        cnt     = contours[i]
        arc     = cv2.arcLength(cnt, True)
        
        # ULTRA-FINE DETAIL: This ensures perfectly curved petals instead of polygons
        epsilon = max(0.05, 0.0005 * arc)       
        
        approx  = cv2.approxPolyDP(cnt, epsilon, closed=True)
        if len(approx) < 2:
            continue

        pts = []
        for pt in approx:
            x_mm = round(pt[0][0] * mm_per_px, 3)
            y_mm = round(h_mm - pt[0][1] * mm_per_px, 3)   # flip Y
            pts.append((x_mm, y_mm))

        # Remove consecutive near-duplicates
        deduped = [pts[0]]
        for p in pts[1:]:
            if _dist(p, deduped[-1]) > 0.05:
                deduped.append(p)

        if len(deduped) >= 2:
            loops.append(deduped)

    if not loops:
        return "; ERROR: No valid loops after simplification."

    # ── 7. Nearest-neighbour ordering ────────────────────────────
    unvisited = list(range(len(loops)))
    ordered   = []
    pen       = (0.0, 0.0)

    while unvisited:
        best_idx   = -1
        best_loop  = None
        best_dist  = float('inf')
        best_entry = 0

        for i in unvisited:
            loop = loops[i]
            for j, pt in enumerate(loop):
                d = _dist(pen, pt)
                if d < best_dist:
                    best_dist  = d
                    best_idx   = i
                    best_entry = j
                    best_loop  = loop

        rotated = best_loop[best_entry:] + best_loop[:best_entry]
        unvisited.remove(best_idx)
        ordered.append(rotated)
        pen = rotated[0]   # after full loop pen is back at entry

    # ── 8. Build G-code ──────────────────────────────────────────
    g = []
    g.append('; ════════════════════════════════════════')
    g.append('; Mehendi Plotter — Exact Replica Smooth Curves')
    g.append(f'; Design  : {design_name}')
    g.append(f'; Scale   : {scale}x  ({round(target_width_mm, 1)} mm wide)')
    g.append(f'; Shapes  : {len(ordered)}')
    g.append('; ════════════════════════════════════════')
    g.append('G21')         # mm
    g.append('G90')         # absolute
    g.append('G92 X0 Y0')   # set current position as origin

    for i, loop in enumerate(ordered):
        g.append(f'; ── Shape {i+1}/{len(ordered)} ({len(loop)} pts) ──')
        # Travel to entry point at fast speed (continuous path, no lifts)
        g.append(f'G1 F{FEED_TRAVEL} X{loop[0][0]:.3f} Y{loop[0][1]:.3f}')
        # Draw the curve perfectly
        g.append(f'G1 F{FEED_DRAW}')
        for pt in loop[1:]:
            g.append(f'G1 X{pt[0]:.3f} Y{pt[1]:.3f}')
        # Close back to entry
        g.append(f'G1 X{loop[0][0]:.3f} Y{loop[0][1]:.3f}')

    g.append('; ── Job complete ──')
    g.append(f'G1 F{FEED_TRAVEL} X0.000 Y0.000')
    g.append('; ════ END ════')

    return '\n'.join(g)