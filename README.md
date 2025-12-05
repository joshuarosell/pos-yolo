# POS3: Vision-based Point-of-Sale

This app uses a YOLO ONNX model (`best.onnx`) and dataset config (`data.yaml`) to recognize grocery items in real time (< 1s latency), draw bounding boxes on a live camera feed, tally a virtual receipt, play an audible signal upon successful recognition, detect open/closed-hand gestures to start/end the customer session, and show the total due at the end.

## Quick Start

1. Ensure the virtual environment is active:
   - Windows PowerShell:
     ```powershell
     .\pos3_env\Scripts\Activate.ps1
     ```
2. Install dependencies (already added to `requirements.txt`):
   ```powershell
   "D:/ARTIFICIAL INTELLIGENCE/AI 231 MLOPS/POS3/pos3_env/Scripts/python.exe" -m pip install -r requirements.txt
   ```
3. Run the app:
   ```powershell
   "D:/ARTIFICIAL INTELLIGENCE/AI 231 MLOPS/POS3/pos3_env/Scripts/python.exe" app.py
   ```

## Usage
- Show an open hand to start a session (the receipt resets).
- Present one item at a time; on recognition, a beep plays and the item is added to the receipt.
- Show a closed fist to end the session; the total due is displayed.

## Notes
- Prices are configurable in `prices.yaml`. If missing, default price of 1.00 is used.
- Latency depends on camera resolution and model; reduce `--width`/`--height` to improve performance if needed.
