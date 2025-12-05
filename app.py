import os
import sys
import time
import yaml
import math
import queue
import threading
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import numpy as np
import onnxruntime as ort

# On Windows, winsound provides a simple beep
try:
    import winsound
    def beep_success():
        winsound.Beep(1200, 120)
except Exception:
    def beep_success():
        pass

# Mediapipe for hand gesture detection (open vs closed hand)
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
except Exception:
    mp_hands = None

@dataclass
class Detection:
    cls_id: int
    score: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2

class YOLOOnnxDetector:
    def __init__(self, onnx_path: str, class_names: List[str], conf_threshold: float = 0.25, iou_threshold: float = 0.45, input_size: Tuple[int, int] = (640, 640)):
        self.onnx_path = onnx_path
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_w, self.input_h = input_size
        self.session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])  # CPU for portability
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, float, Tuple[int,int]]:
        h, w = img_bgr.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img_bgr, (nw, nh))
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        top = (self.input_h - nh) // 2
        left = (self.input_w - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized
        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img, scale, left, top

    def postprocess(self, preds: np.ndarray, scale: float, left: int, top: int, orig_shape: Tuple[int,int]) -> List[Detection]:
        # Expect YOLOv5/8 ONNX format: [batch, num, 85] -> [x,y,w,h,conf,cls_probs]
        if preds.ndim == 3:
            preds = preds[0]
        boxes = preds[:, :4]
        scores = preds[:, 4]
        class_probs = preds[:, 5:]
        class_ids = np.argmax(class_probs, axis=1)
        class_scores = class_probs[np.arange(len(class_probs)), class_ids]
        conf = scores * class_scores
        mask = conf > self.conf_threshold
        boxes = boxes[mask]
        conf = conf[mask]
        class_ids = class_ids[mask]
        # Convert from x,y,w,h to x1,y1,x2,y2 in original image scale
        H, W = orig_shape
        result = []
        if boxes.size == 0:
            return result
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - bw/2)
        y1 = (cy - bh/2)
        x2 = (cx + bw/2)
        y2 = (cy + bh/2)
        # Undo letterbox
        x1 = (x1 - left) / scale
        y1 = (y1 - top) / scale
        x2 = (x2 - left) / scale
        y2 = (y2 - top) / scale
        # Clip
        x1 = np.clip(x1, 0, W-1)
        y1 = np.clip(y1, 0, H-1)
        x2 = np.clip(x2, 0, W-1)
        y2 = np.clip(y2, 0, H-1)
        # NMS
        dets = np.stack([x1, y1, x2, y2, conf, class_ids], axis=1)
        indices = nms(dets, self.iou_threshold)
        for i in indices:
            xi1, yi1, xi2, yi2, c, cls = dets[i]
            result.append(Detection(cls_id=int(cls), score=float(c), bbox=(int(xi1), int(yi1), int(xi2), int(yi2))))
        return result

    def infer(self, img_bgr: np.ndarray) -> List[Detection]:
        H, W = img_bgr.shape[:2]
        blob, scale, left, top = self.preprocess(img_bgr)
        out = self.session.run(self.output_names, {self.input_name: blob})
        preds = out[0]
        return self.postprocess(preds, scale, left, top, (H, W))


def nms(dets: np.ndarray, iou_thr: float) -> List[int]:
    # dets: [N, 6] with [x1,y1,x2,y2,score,cls]
    x1, y1, x2, y2, scores = dets[:,0], dets[:,1], dets[:,2], dets[:,3], dets[:,4]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_rest = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
        iou = inter / (area_i + area_rest - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep

class GestureDetector:
    def __init__(self):
        self.enabled = mp_hands is not None
        if self.enabled:
            self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # Hysteresis thresholds on normalized, scale-invariant distances
        self.open_thr = 0.40
        self.closed_thr = 0.25
        from collections import deque
        self.history = deque(maxlen=5)
        self.last_stable = "none"
        self.last_metric = 0.0

    def _classify_raw(self, frame_bgr: np.ndarray) -> Tuple[str, float]:
        if not self.enabled:
            return "unknown", 0.0
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(frame_rgb)
        if not res.multi_hand_landmarks:
            return "none", 0.0
        lm = res.multi_hand_landmarks[0]
        pts = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)  # normalized [0,1]
        # Palm approximation: mean of wrist and MCPs
        palm_idx = [0, 1, 5, 9, 13, 17]
        palm = pts[palm_idx].mean(axis=0)
        tips_idx = [4, 8, 12, 16, 20]
        tips = pts[tips_idx]
        # Scale based on hand bbox to make distance scale-invariant
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        scale = float(max(max_xy - min_xy)) + 1e-6
        d = float(np.linalg.norm(tips - palm, axis=1).mean() / scale)
        if d > self.open_thr:
            return "open", d
        if d < self.closed_thr:
            return "closed", d
        return "neutral", d

    def classify(self, frame_bgr: np.ndarray) -> str:
        label, metric = self._classify_raw(frame_bgr)
        self.last_metric = metric
        self.history.append(label)
        # Majority vote over recent frames, prefer non-neutral
        if len(self.history) < self.history.maxlen:
            return label
        vals, counts = np.unique(list(self.history), return_counts=True)
        majority = vals[np.argmax(counts)]
        if majority != "neutral" and majority != "none":
            self.last_stable = majority
            return majority
        return self.last_stable if self.last_stable else label

class Receipt:
    def __init__(self, class_names: List[str], prices: Dict[str, float]):
        self.class_names = class_names
        self.prices = prices
        self.items: List[str] = []
        self.total = 0.0

    def add(self, cls_id: int):
        name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else f"class_{cls_id}"
        price = self.prices.get(name, 1.0)
        self.items.append(name)
        self.total += price

    def reset(self):
        self.items.clear()
        self.total = 0.0

# Draw helpers

def draw_detections(frame: np.ndarray, detections: List[Detection], class_names: List[str]):
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f"{class_names[d.cls_id]} {d.score:.2f}"
        cv2.putText(frame, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)


def overlay_receipt(frame: np.ndarray, receipt: Receipt, state: str):
    panel_w = 320
    h, w = frame.shape[:2]
    x0 = w - panel_w
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 0), (w, h), (0,0,0), -1)
    alpha = 0.4
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
    y = 24
    cv2.putText(frame, f"Session: {state}", (x0+12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    y += 28
    cv2.putText(frame, "Items:", (x0+12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    y += 24
    for name in receipt.items[-12:]:
        cv2.putText(frame, f"- {name}", (x0+12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        y += 22
    y = h - 48
    cv2.putText(frame, f"Total: ${receipt.total:.2f}", (x0+12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)


def load_data_yaml(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    # Expect 'names' field
    names = data.get('names')
    if isinstance(names, dict):
        # Convert dict {id:name}
        names = [names[i] for i in sorted(names.keys())]
    if not isinstance(names, list):
        raise ValueError("data.yaml missing 'names' list")
    return names


def load_prices(path: str, class_names: List[str]) -> Dict[str, float]:
    if not os.path.exists(path):
        return {name: 1.0 for name in class_names}
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    prices = {}
    for name in class_names:
        prices[name] = float(data.get(name, 1.0))
    return prices


def main():
    parser = argparse.ArgumentParser(description='POS3 - Vision-based POS using YOLO ONNX')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (USB cameras often 1 or 2)')
    parser.add_argument('--width', type=int, default=640, help='Capture width')
    parser.add_argument('--height', type=int, default=480, help='Capture height')
    parser.add_argument('--fps', type=int, default=30, help='Capture FPS target')
    parser.add_argument('--backend', choices=['auto','dshow','msmf'], default='auto', help='Video backend on Windows')
    args = parser.parse_args()
    onnx_path = os.path.join(os.getcwd(), 'best.onnx')
    yaml_path = os.path.join(os.getcwd(), 'data.yaml')
    price_path = os.path.join(os.getcwd(), 'prices.yaml')
    if not os.path.exists(onnx_path):
        print('best.onnx not found')
        return
    if not os.path.exists(yaml_path):
        print('data.yaml not found')
        return
    class_names = load_data_yaml(yaml_path)
    prices = load_prices(price_path, class_names)

    detector = YOLOOnnxDetector(onnx_path, class_names, conf_threshold=0.35, iou_threshold=0.5, input_size=(640, 640))
    gest = GestureDetector()
    receipt = Receipt(class_names, prices)

    backend_flag = None
    if args.backend == 'dshow':
        backend_flag = cv2.CAP_DSHOW
    elif args.backend == 'msmf':
        backend_flag = cv2.CAP_MSMF

    cap = cv2.VideoCapture(args.camera, backend_flag) if backend_flag is not None else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'Camera index {args.camera} failed to open. Scanning fallbacks...')
        opened = False
        for idx in range(0, 6):
            tmp = cv2.VideoCapture(idx, backend_flag) if backend_flag is not None else cv2.VideoCapture(idx)
            time.sleep(0.1)
            if tmp.isOpened():
                cap = tmp
                opened = True
                print(f'Opened camera index {idx}.')
                break
            tmp.release()
        if not opened:
            print('No available camera found. Plug in a USB camera and retry with --camera N.')
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    session_state = 'idle'  # idle|active|ended
    last_beep_time = 0.0
    recognized_cache = set()

    print('Press Q to quit.')
    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            break

        # Gesture-based session control
        gesture = gest.classify(frame)
        if gesture == 'open' and session_state != 'active':
            session_state = 'active'
            receipt.reset()
            recognized_cache.clear()
        elif gesture == 'closed' and session_state == 'active':
            session_state = 'ended'

        detections: List[Detection] = detector.infer(frame)
        draw_detections(frame, detections, class_names)

        # When session active, add first high-confidence detection; items one-by-one
        if session_state == 'active':
            if detections:
                # Take top by score
                det = max(detections, key=lambda d: d.score)
                name = class_names[det.cls_id]
                # Add only once per item name until gesture open again or item changes
                if name not in recognized_cache and det.score > 0.5:
                    receipt.add(det.cls_id)
                    recognized_cache.add(name)
                    if time.time() - last_beep_time > 0.15:
                        beep_success()
                        last_beep_time = time.time()

        overlay_receipt(frame, receipt, session_state)

        if session_state == 'ended':
            cv2.putText(frame, 'Session Ended', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.putText(frame, f'Total Due: ${receipt.total:.2f}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        # Show gesture debug
        cv2.putText(frame, f"Gesture: {gesture} {gest.last_metric:.2f}", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # Show latency
        dt = time.time() - t0
        cv2.putText(frame, f"Latency: {dt*1000:.1f} ms", (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow('POS3 - Vision POS', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
