import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import torch
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scripts.model import CNN
from scripts.data_loader import get_dataloaders
from scripts.trainer import train_model
from scripts.inference import predict_image

class ChestMNISTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ChestMNIST Classifier")
        self.root.geometry("900x700")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.history = None
        self.training_thread = None
        self.stop_event = threading.Event()
        self.image_path = None

        self.setup_ui()
        self.log(f"Device: {self.device}")

    def setup_ui(self):
        left_panel = ttk.Frame(self.root, padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left_panel, text="Training Configuration", font=("Helvetica", 12, "bold")).pack(pady=5)

        ttk.Label(left_panel, text="Batch Size:").pack(anchor=tk.W)
        self.batch_size_var = tk.StringVar(value="32")
        ttk.Entry(left_panel, textvariable=self.batch_size_var).pack(fill=tk.X, pady=2)

        ttk.Label(left_panel, text="Learning Rate:").pack(anchor=tk.W)
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(left_panel, textvariable=self.lr_var).pack(fill=tk.X, pady=2)

        ttk.Label(left_panel, text="Epochs:").pack(anchor=tk.W)
        self.epochs_var = tk.StringVar(value="5")
        ttk.Entry(left_panel, textvariable=self.epochs_var).pack(fill=tk.X, pady=2)

        ttk.Label(left_panel, text="Optimizer:").pack(anchor=tk.W)
        self.optimizer_var = tk.StringVar(value="Adam")
        self.optimizer_combo = ttk.Combobox(left_panel, textvariable=self.optimizer_var, values=["Adam", "SGD", "RMSprop"], state="readonly")
        self.optimizer_combo.pack(fill=tk.X, pady=2)

        ttk.Label(left_panel, text="Criterion:").pack(anchor=tk.W)
        self.criterion_var = tk.StringVar(value="BCEWithLogitsLoss")
        self.criterion_combo = ttk.Combobox(left_panel, textvariable=self.criterion_var, values=["BCEWithLogitsLoss", "CrossEntropyLoss", "MSELoss"], state="readonly")
        self.criterion_combo.pack(fill=tk.X, pady=2)

        self.train_btn = ttk.Button(left_panel, text="Start Training", command=self.start_training)
        self.train_btn.pack(pady=10, fill=tk.X)
        
        self.stop_btn = ttk.Button(left_panel, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(pady=2, fill=tk.X)

        self.load_model_btn = ttk.Button(left_panel, text="Load Model", command=self.load_model)
        self.load_model_btn.pack(pady=2, fill=tk.X)

        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(left_panel, text="Inference", font=("Helvetica", 12, "bold")).pack(pady=5)
        
        self.select_img_btn = ttk.Button(left_panel, text="Select Image", command=self.select_image)
        self.select_img_btn.pack(pady=5, fill=tk.X)

        self.predict_btn = ttk.Button(left_panel, text="Predict", command=self.predict, state=tk.DISABLED)
        self.predict_btn.pack(pady=5, fill=tk.X)

        right_panel = ttk.Frame(self.root, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(right_panel, text="Logs:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        self.log_text = tk.Text(right_panel, height=15, state=tk.DISABLED, bg="#f0f0f0")
        self.log_text.pack(fill=tk.X, pady=5)
        
        self.image_label = ttk.Label(right_panel, text="No Image Selected")
        self.image_label.pack(pady=10)

        ttk.Label(right_panel, text="Prediction Results:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        self.result_text = tk.Text(right_panel, height=10, state=tk.DISABLED, bg="#ffffff")
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=5)

    def log(self, message):
        self.root.after(0, self._log_internal, message)

    def _log_internal(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)


    def load_model(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                self.log(f"Loading model from {file_path}...")
                self.model = CNN().to(self.device)
                self.model.load_state_dict(torch.load(file_path, map_location=self.device))
                self.model.eval()
                self.log("Model loaded successfully!")
                self.predict_btn.config(state=tk.NORMAL)
            except Exception as e:
                self.log(f"Error loading model: {str(e)}")
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

    def start_training(self):
        try:
            batch_size = int(self.batch_size_var.get())
            lr = float(self.lr_var.get())
            epochs = int(self.epochs_var.get())
            optimizer_name = self.optimizer_var.get()
            criterion_name = self.criterion_var.get()
        except ValueError:
            messagebox.showerror("Error", "Invalid configuration values.")
            return

        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.predict_btn.config(state=tk.DISABLED)
        self.stop_event.clear()

        self.training_thread = threading.Thread(
            target=self.run_training,
            args=(batch_size, lr, epochs, optimizer_name, criterion_name),
            daemon=True
        )
        self.training_thread.start()

    def stop_training(self):
        if self.training_thread and self.training_thread.is_alive():
            self.stop_event.set()
            self.log("Stopping training...")

    def run_training(self, batch_size, lr, epochs, optimizer_name, criterion_name):
        try:
            self.log("Loading data (this might take a while if downloading)...")
            train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)
            
            self.log("Initializing model...")
            self.model = CNN().to(self.device)
            
            self.log(f"Starting training for {epochs} epochs with {optimizer_name} and {criterion_name}...")
            self.model, self.history = train_model(
                self.model, 
                train_loader, 
                val_loader, 
                self.device, 
                num_epochs=epochs, 
                lr=lr,
                log_callback=self.log,
                stop_event=self.stop_event,
                optimizer_name=optimizer_name,
                criterion_name=criterion_name
            )
            
            if not self.stop_event.is_set():
                self.log("Training completed successfully!")
            else:
                self.log("Training stopped.")

        except Exception as e:
            self.log(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.root.after(0, self.training_finished)

    def training_finished(self):
        self.train_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if self.model:
            self.predict_btn.config(state=tk.NORMAL)
        if self.history:
            self.show_training_plot()

    def show_training_plot(self):
        window = tk.Toplevel(self.root)
        window.title("Training History")
        window.geometry("800x600")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        ax1.plot(epochs, self.history['train_loss'], label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], label='Val Loss')
        ax1.set_title('Loss vs Epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        if any(self.history['val_acc']):
             ax2.plot(epochs, self.history['val_acc'], label='Val Acc')
        if any(self.history['val_auc']):
             ax2.plot(epochs, self.history['val_auc'], label='Val AUC')
        
        ax2.set_title('Metrics vs Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.log(f"Selected image: {os.path.basename(file_path)}")
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.config(state=tk.DISABLED)

    def display_image(self, file_path):
        try:
            img = Image.open(file_path)
            img.thumbnail((250, 250))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            self.log(f"Error loading image: {e}")

    def predict(self):
        if not self.model:
            messagebox.showwarning("Warning", "Model is not trained yet!")
            return
        if not self.image_path:
            messagebox.showwarning("Warning", "No image selected!")
            return

        self.log("Predicting...")
        try:
            results = predict_image(self.model, self.image_path, self.device)
            if "error" in results:
                self.log(f"Prediction error: {results['error']}")
            else:
                self.show_results(results)
                self.log("Prediction done.")
        except Exception as e:
            self.log(f"Error: {e}")

    def show_results(self, results):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        for class_name, prob in sorted_results:
            self.result_text.insert(tk.END, f"{class_name}: {prob:.4f}\n")
            
        self.result_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChestMNISTApp(root)
    root.mainloop()
