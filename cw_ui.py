import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import cw_code  

class App:
    def __init__(self, root):
        self.root = root
        root.title('Video Panorama Generator')

        # Set up video path input
        ttk.Label(root, text="Video Path:").grid(column=0, row=0)
        self.video_path_entry = ttk.Entry(root, width=50)
        self.video_path_entry.grid(column=1, row=0, sticky='we')
        ttk.Button(root, text="Browse", command=self.load_file).grid(column=2, row=0)

        # Set up motion threshold input
        ttk.Label(root, text="Motion Threshold:").grid(column=0, row=1)
        self.motion_threshold_entry = ttk.Entry(root, width=10)
        self.motion_threshold_entry.grid(column=1, row=1, sticky='w')
        self.motion_threshold_entry.insert(0, "0.8")

        # Set up skip frames input
        ttk.Label(root, text="Skip Frames:").grid(column=0, row=2)
        self.skip_frames_entry = ttk.Entry(root, width=10)
        self.skip_frames_entry.grid(column=1, row=2, sticky='w')
        self.skip_frames_entry.insert(0, "10")

        # Options checkboxes
        self.apply_sharpen = tk.BooleanVar()
        ttk.Checkbutton(root, text="Apply Sharpen", variable=self.apply_sharpen).grid(column=0, row=3)
        self.apply_motion_filter = tk.BooleanVar()
        ttk.Checkbutton(root, text="Apply Motion Filter", variable=self.apply_motion_filter).grid(column=1, row=3)

        # Set up output filename input
        ttk.Label(root, text="Output Filename:").grid(column=0, row=4)
        self.output_name_entry = ttk.Entry(root, width=50)
        self.output_name_entry.grid(column=1, row=4, sticky='we')
        self.output_name_entry.insert(0, "panorama.jpg")

        # Run button
        ttk.Button(root, text="Generate Panorama", command=self.start_thread).grid(column=0, row=5, columnspan=3)

        # Progress bar
        self.progress = ttk.Progressbar(root, length=100, mode='indeterminate')
        self.progress.grid(column=0, row=6, columnspan=3, sticky='we')

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[
            ("All Supported Formats", "*.avi;*.mp4;*.mov;*.mkv;*.flv;*.wmv"),
            ("AVI Files", "*.avi"),
            ("MP4 Files", "*.mp4"),
            ("MOV Files", "*.mov"),
            ("MKV Files", "*.mkv"),
            ("FLV Files", "*.flv"),
            ("WMV Files", "*.wmv"),
            ("All Files", "*.*")
        ])
        if file_path:
            self.video_path_entry.delete(0, tk.END)
            self.video_path_entry.insert(0, file_path)

    def start_thread(self):
        self.progress.start()
        thread = threading.Thread(target=self.generate_panorama)
        thread.start()

    def generate_panorama(self):
        video_path = self.video_path_entry.get()
        motion_threshold = float(self.motion_threshold_entry.get())
        skip_frames = int(self.skip_frames_entry.get())
        output_name = self.output_name_entry.get()
        cw_code.main(video_path, apply_sharpen=self.apply_sharpen.get(),
                     apply_motion_filter=self.apply_motion_filter.get(),
                     motion_threshold=motion_threshold,
                     skip_frames=skip_frames, output_name=output_name)
        self.progress.stop()
        messagebox.showinfo("Complete", "Panorama generation completed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
