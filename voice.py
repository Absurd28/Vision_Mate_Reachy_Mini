import speech_recognition as sr
import threading
import queue

class ReachyVoice:
    """
    Voice Listener for Reachy Mini with Phonetic Fallback.
    Handles common speech-to-text misinterpretations.
    """
    
    # PHONETIC FALLBACK DICTIONARY
    # Maps common misheard phrases to correct robot commands
    PHONETIC_MAP = {
        "gulab ka botal": "grab the bottle",
        "reaching": "reachy",
        "trichy": "reachy",
        "witchy": "reachy",
        "richard": "reachy",
        "fetch the hotel": "fetch the bottle",
        "catch the bottle": "grab the bottle",
        "get the medal": "get the bottle"
    }

    def __init__(self, cmd_queue, wake_word="reachy"):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.cmd_queue = cmd_queue 
        self.wake_word = wake_word
        self.is_listening = False

    def listen_loop(self, log_callback):
        self.is_listening = True
        log_callback("Voice: Listener Active. Correction Engine Loaded.")
        
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while self.is_listening:
                try:
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)
                    raw_text = self.recognizer.recognize_google(audio).lower()
                    
                    # Apply Phonetic Corrections
                    processed_text = raw_text
                    for misheard, correct in self.PHONETIC_MAP.items():
                        if misheard in processed_text:
                            print(f"[Voice Thread] Correcting phonetic error: '{misheard}' -> '{correct}'")
                            processed_text = processed_text.replace(misheard, correct)
                    
                    print(f"[Voice Thread] Final Text: '{processed_text}'")
                    log_callback(f"Heard: '{processed_text}'")
                    
                    # If wake word or direct command after correction
                    if self.wake_word in processed_text or any(cmd in processed_text for cmd in ["bottle", "stop", "move"]):
                        self.cmd_queue.put(processed_text)
                        
                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    continue
                except Exception as e:
                    print(f"[Voice Error] {e}")

    def start(self, command_callback, log_callback=None):
        # Note: We use the provided callback (usually main.process_with_ai)
        threading.Thread(target=self.listen_loop, args=(command_callback,), daemon=True).start()

    def stop(self):
        self.is_listening = False
