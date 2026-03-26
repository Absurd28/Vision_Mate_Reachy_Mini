import cv2

class ReachyVision:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret: return None
        
        # --- PLACEHOLDER FOR EYE TRACKING PIPELINE ---
        # cursor_x, cursor_y = eye_tracker.get_gaze()
        # cv2.circle(frame, (cursor_x, cursor_y), 10, (0, 255, 0), 2)
        
        cv2.putText(frame, "REACHY VISION ACTIVE", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        self.cap.release()
