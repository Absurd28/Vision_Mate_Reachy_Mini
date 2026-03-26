import threading
import queue
from ui import ReachyDashboard
from vision import ReachyVision
from voice import ReachyVoice
from robot import ReachyRobot
from nlp_inference import CommandClassifier

class VisionMateApp:
    """
    Orchestrates the VisionMate AI platform.
    Now with integrated DistilBERT Intent Classification.
    """
    def __init__(self):
        # 1. Initialize Thread-Safe Queue
        self.robot_queue = queue.Queue()
        
        # 2. Initialize UI
        self.ui = ReachyDashboard(self.start_system, self.stop_system)
        
        # 3. Initialize AI Brain
        try:
            self.nlp = CommandClassifier()
            self.ui.log("System: AI Intent Engine Loaded.")
        except Exception as e:
            self.ui.log(f"System Error: Failed to load AI model. {e}")
            self.nlp = None
        
        self.vision = None
        self.voice = None
        self.robot = None
        self.is_running = False

    def start_system(self):
        if self.is_running: return
        self.is_running = True
        
        self.ui.log("System: Powering up modules...")

        # Initialize hardware/background modules
        self.vision = ReachyVision()
        self.voice = ReachyVoice(self.robot_queue)
        
        # We override the voice listener callback to pass through the AI bridge
        self.voice.start(self.process_with_ai)
        
        self.robot = ReachyRobot(self.robot_queue)
        self.robot.start(self.ui.log)
        
        self.ui.log("System: All threads active. Ready for commands.")
        self.update_loop()

    def process_with_ai(self, raw_text):
        """
        AI Bridge: Transcribed text -> DistilBERT -> Robot Intent.
        Now with verbose logging for debugging.
        """
        if self.nlp:
            # 1. Run inference
            intent, confidence = self.nlp.predict_intent(raw_text)
            
            # 2. VERBOSE LOGGING
            log_msg = f"[NLP ENGINE] Classified Intent: {intent} (Confidence: {confidence:.2f})"
            print(log_msg)
            self.ui.log(f"Heard: '{raw_text}'")
            self.ui.log(log_msg)
            
            # 3. Action: Only pass to robot if we are reasonably confident
            if confidence > 0.4:
                self.robot_queue.put(intent)
            else:
                self.ui.log("System: Confidence too low, ignoring.")
        else:
            self.robot_queue.put(raw_text)

    def stop_system(self):
        self.is_running = False
        if self.vision: self.vision.release()
        if self.voice: self.voice.stop()
        if self.robot: self.robot.stop()
        self.ui.log("System: Powered down.")

    def update_loop(self):
        if not self.is_running: return
        frame = self.vision.get_frame()
        if frame is not None:
            self.ui.update_video(frame)
        self.ui.after(33, self.update_loop)

if __name__ == "__main__":
    app = VisionMateApp()
    app.ui.mainloop()
