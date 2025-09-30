"""
gui.py - User Interface
Only handles user interface logic, all business logic is delegated to controller
"""

import tkinter as tk
import threading
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os


class LoadingDialog:
    """Loading Dialog"""
    
    def __init__(self, parent):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Processing...")
        self.dialog.geometry("250x100")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center display
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 75, parent.winfo_rooty() + 75))
        
        # Loading label
        tk.Label(self.dialog, text="Loading...", font=('Arial', 11)).pack(pady=15)
        
        # Progress bar
        style = ttk.Style()
        style.configure("Custom.Horizontal.TProgressbar")
        
        self.progress = ttk.Progressbar(
            self.dialog, 
            mode='indeterminate', 
            length=150,
            style="Custom.Horizontal.TProgressbar"
        )
        self.progress.pack(pady=5)
        self.progress.start(8)
        
        # Prevent closing
        self.dialog.protocol("WM_DELETE_WINDOW", lambda: None)
        
    def destroy(self):
        """Close dialog"""
        self.progress.stop()
        self.dialog.grab_release()
        self.dialog.destroy()


class LifetimeGUI:
    """Lifetime Analysis GUI - Only handles interface logic"""
    
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller  # Main controller
        self.root.title("Lifetime Analysis GUI")
        
        self.file_path = ""
        self.loading_dialog = None
        self.analysis_running = False
        
        self._setup_ui()
        self._setup_bindings()
        
    def _setup_ui(self):
        """Setup user interface"""
        # Add header section
        self._add_header_section()
        
        # File selection
        tk.Label(self.root, text="Input File:").grid(row=2, column=0, sticky="e")
        self.file_entry = tk.Entry(self.root, width=40)
        self.file_entry.grid(row=2, column=1, padx=5, pady=5)
        self.load_button = tk.Button(self.root, text="Browse", command=self.load_file)
        self.load_button.grid(row=2, column=2, padx=5, pady=5)

        # Lifecycle input
        tk.Label(self.root, text="Expected Lifecycle (cycles):").grid(row=3, column=0, sticky="e")
        self.lifecycle_entry = tk.Entry(self.root, width=10)
        self.lifecycle_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)

        # Run button
        self.run_button = tk.Button(self.root, text="Run Analysis", command=self.start_analysis)
        self.run_button.grid(row=4, column=1, pady=15)
        
    def _add_header_section(self):
        """Add simple header section"""
        # Remove Amazon icon and Product Integrity Team references
        pass
        
    def _setup_bindings(self):
        """Setup keyboard bindings"""
        self.file_entry.bind('<Return>', self._on_enter_pressed)
        self.lifecycle_entry.bind('<Return>', self._on_enter_pressed)
        self.root.bind('<Return>', self._on_enter_pressed)
        self.file_entry.focus_set()

    def _on_enter_pressed(self, event):
        """Handle Enter key press"""
        self.start_analysis()

    def load_file(self):
        """Load file dialog"""
        path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[
                ("All supported", "*.csv;*.txt;*.xlsx"),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.file_path = path
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, path)

    def start_analysis(self):
        """Start analysis - UI validation and thread management"""
        # Input validation
        if not self._validate_inputs():
            return
            
        # Prevent duplicate execution
        if self.analysis_running:
            return
            
        self.analysis_running = True
        
        # Show loading dialog
        self.loading_dialog = LoadingDialog(self.root)
        self.run_button.config(state='disabled')
        
        # Execute analysis in new thread
        expected_cycles = int(self.lifecycle_entry.get())
        analysis_thread = threading.Thread(
            target=self._run_analysis_thread, 
            args=(expected_cycles,)
        )
        analysis_thread.daemon = True
        analysis_thread.start()

    def _validate_inputs(self):
        """Validate user inputs"""
        if not self.file_path:
            messagebox.showerror("Error", "Please select a data file.")
            return False

        if not self.file_path or not self.file_path.strip():
            messagebox.showerror("Error", "Please select a valid file.")
            return False

        # Check file format
        import os
        if not os.path.exists(self.file_path):
            messagebox.showerror("Error", "Selected file does not exist.")
            return False
            
        ext = os.path.splitext(self.file_path)[-1].lower()
        if ext not in ['.csv', '.txt', '.xlsx']:
            messagebox.showerror("Error", "Unsupported file format. Please select a CSV, TXT, or Excel file.")
            return False

        try:
            int(self.lifecycle_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid expected lifecycle number.")
            return False
            
        return True

    def _run_analysis_thread(self, expected_cycles):
        """Run analysis in separate thread - delegate to controller"""
        try:
            # Delegate all business logic to controller
            figures, text_output, success, error_message = self.controller.load_and_analyze_data(
                self.file_path, expected_cycles
            )
            
            if success:
                # Generate PDF report
                pdf_path, report_error = self.controller.generate_report(figures, text_output)
                
                if pdf_path:
                    # Update UI in main thread
                    self.root.after(0, self._analysis_completed, pdf_path)
                else:
                    self.root.after(0, self._analysis_failed, report_error or "Failed to generate report")
            else:
                self.root.after(0, self._analysis_failed, error_message)
                
        except Exception as e:
            self.root.after(0, self._analysis_failed, str(e))

    def _analysis_completed(self, pdf_path):
        """UI update when analysis completed"""
        self._reset_ui_state()
        
        # Automatically open PDF
        success = self.controller.open_pdf_file(pdf_path)
        if not success:
            messagebox.showwarning(
                "Warning", 
                f"Analysis completed but could not open PDF automatically.\nFile saved at: {pdf_path}"
            )

    def _analysis_failed(self, error_message):
        """UI update when analysis failed"""
        self._reset_ui_state()
        messagebox.showerror("Error during analysis", f"Error details: {error_message}")

    def _reset_ui_state(self):
        """Reset UI state"""
        self.analysis_running = False
        
        if self.loading_dialog:
            self.loading_dialog.destroy()
            self.loading_dialog = None
            
        self.run_button.config(state='normal')
