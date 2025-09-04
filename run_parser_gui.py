import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import subprocess
import os
import threading
import glob
import sys
import logging
import expense_parser_all

# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join("logs", "parser.log"),
    filemode="a",  # append to existing log
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Determine base path
if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

REPORTS_DIR = os.path.join(base_path, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

from expense_parser_all import main as run_parser_main

# ------------------- Functions -------------------
def run_parser_thread():
    threading.Thread(target=run_parser).start()

# Set OpenAI API key for the parser
os.environ["OPENAI_API_KEY"] = "REDACTED"

def run_parser():
    try:
        log_text.config(state='normal')
        log_text.delete('1.0', tk.END)
        log_text.insert(tk.END, "🚀 Starting parser...\n")
        log_text.update()
        logging.info("🚀 Starting parser...")

        # Get PDF folder from environment variable or default
        pdf_folder = os.environ.get("PDF_DIR", "c:/cca/data")
        reports_folder = REPORTS_DIR

        # Call parser function
        run_parser_main(pdf_dir_override=pdf_folder, reports_dir_override=reports_folder)

        # Refresh reports list after parsing
        refresh_report_lists()
        log_text.insert(tk.END, "\n✅ Parser finished successfully!\n")
        logging.info("✅ Parser finished successfully!")
        messagebox.showinfo("Success", "Parser finished! Reports are in the /reports folder.")

    except Exception as e:
        log_text.insert(tk.END, f"\n❌ Parser failed: {str(e)}\n")
        log_text.update()
        logging.error("Parser failed: %s", str(e))
        messagebox.showerror("Error", str(e))

def select_data_folder():
    folder = filedialog.askdirectory()
    if folder:
        os.environ["PDF_DIR"] = folder
        data_folder_label.config(text=f"Data Folder: {folder}")

def open_reports_folder():
    folder = REPORTS_DIR
    if os.path.exists(folder):
        os.startfile(folder)
    else:
        messagebox.showerror("Error", f"Reports folder not found: {folder}")
        logging.error(f"❌ Report folder not found: {folder}")
        print(f"❌ Report folder not found: {folder}")

def open_selected_file(file_type, dropdown):
    selection = dropdown.get()
    if selection:
        file_path = os.path.join(REPORTS_DIR, selection)
        if os.path.exists(file_path):
            os.startfile(file_path)
        else:
            messagebox.showerror("Error", f"{selection} not found.")

def refresh_report_lists():
    folder = REPORTS_DIR
    types = {"Excel": "*.xlsx", "PDF": "*_REPORT.pdf", "ZIP": "*.zip"}
    for key, pattern in types.items():
        files = sorted(glob.glob(os.path.join(folder, pattern)), key=os.path.getmtime, reverse=True)
        files_names = [os.path.basename(f) for f in files]
        if key == "Excel":
            excel_dropdown['values'] = files_names
            if files_names:
                excel_dropdown.current(0)
        elif key == "PDF":
            pdf_dropdown['values'] = files_names
            if files_names:
                pdf_dropdown.current(0)
        elif key == "ZIP":
            zip_dropdown['values'] = files_names
            if files_names:
                zip_dropdown.current(0)

# ------------------- GUI -------------------
root = tk.Tk()
root.title("Expense Parser - One Click")
root.geometry("700x600")

tk.Label(root, text="📂 Select PDF Data Folder (optional)").pack(pady=5)
data_folder_label = tk.Label(root, text="Using default: c:/data")
data_folder_label.pack(pady=2)
tk.Button(root, text="Choose Folder", command=select_data_folder).pack(pady=5)

tk.Button(root, text="Run Parser", command=run_parser_thread, height=2, width=25, bg="#4CAF50", fg="white").pack(pady=10)

tk.Label(root, text="Progress").pack(pady=2)
progress = ttk.Progressbar(root, length=550, mode='determinate')
progress.pack(pady=5)

tk.Label(root, text="Log Output").pack(pady=2)
log_text = tk.Text(root, height=15, width=80, state='disabled')
log_text.pack(pady=5)

tk.Button(root, text="Open Reports Folder", command=open_reports_folder, height=1, width=30, bg="#2196F3", fg="white").pack(pady=5)

frame_dropdowns = tk.Frame(root)
frame_dropdowns.pack(pady=10)

tk.Label(frame_dropdowns, text="Excel Reports").grid(row=0, column=0, padx=5)
excel_dropdown = ttk.Combobox(frame_dropdowns, width=60)
excel_dropdown.grid(row=0, column=1, padx=5)
tk.Button(frame_dropdowns, text="Open", command=lambda: open_selected_file("Excel", excel_dropdown), bg="#FF9800", fg="white").grid(row=0, column=2, padx=5)

tk.Label(frame_dropdowns, text="PDF Reports").grid(row=1, column=0, padx=5)
pdf_dropdown = ttk.Combobox(frame_dropdowns, width=60)
pdf_dropdown.grid(row=1, column=1, padx=5)
tk.Button(frame_dropdowns, text="Open", command=lambda: open_selected_file("PDF", pdf_dropdown), bg="#9C27B0", fg="white").grid(row=1, column=2, padx=5)

tk.Label(frame_dropdowns, text="ZIP Reports").grid(row=2, column=0, padx=5)
zip_dropdown = ttk.Combobox(frame_dropdowns, width=60)
zip_dropdown.grid(row=2, column=1, padx=5)
tk.Button(frame_dropdowns, text="Open", command=lambda: open_selected_file("ZIP", zip_dropdown), bg="#F44336", fg="white").grid(row=2, column=2, padx=5)

refresh_report_lists()
root.mainloop()
