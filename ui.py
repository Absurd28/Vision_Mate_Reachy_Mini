import customtkinter as ctk
from PIL import Image, ImageTk

class ReachyDashboard(ctk.CTk):
    def __init__(self, on_start, on_stop):
        super().__init__()
        self.title("VisionMate AI - Reachy Mini Control")
        self.geometry("1100x700")
        ctk.set_appearance_mode("dark")

        # Layout Configuration
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Video Frame Container
        self.video_label = ctk.CTkLabel(self, text="Camera Feed Loading...", width=640, height=480, fg_color="black")
        self.video_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # Sidebar Container
        self.sidebar = ctk.CTkFrame(self, width=300)
        self.sidebar.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.title_label = ctk.CTkLabel(self.sidebar, text="REACHY ASSIST", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(pady=20)

        self.log_box = ctk.CTkTextbox(self.sidebar, width=250, height=400)
        self.log_box.pack(pady=10, padx=10)
        self.log_box.insert("0.0", "System Logs:\n" + "-"*20 + "\n")

        self.btn_start = ctk.CTkButton(self.sidebar, text="Start System", command=on_start, fg_color="#2ecc71", hover_color="#27ae60")
        self.btn_start.pack(pady=10, padx=20, fill="x")

        self.btn_stop = ctk.CTkButton(self.sidebar, text="Stop System", command=on_stop, fg_color="#e74c3c", hover_color="#c0392b")
        self.btn_stop.pack(pady=10, padx=20, fill="x")

    def update_video(self, frame):
        """Updates the video feed label using modern CTkImage for DPI stability."""
        # Convert OpenCV RGB array to PIL Image
        img = Image.fromarray(frame)
        
        # Create a CTkImage object (handles dark/light mode and scaling natively)
        ctk_img = ctk.CTkImage(light_image=img, 
                               dark_image=img, 
                               size=(640, 480))
        
        # Assign to label
        self.video_label.configure(image=ctk_img, text="")
        
        # Keep a reference to prevent garbage collection
        self.video_label._image = ctk_img

    def log(self, message):
        """Thread-safe logging to the UI textbox."""
        self.log_box.insert("end", f"\n{message}")
        self.log_box.see("end")
