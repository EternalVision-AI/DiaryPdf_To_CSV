import os
import cv2
import fitz  # PyMuPDF
import customtkinter as ctk
from tkinter import filedialog, messagebox
from pdf2image import convert_from_path
from diarypage_detection import DetectDiagyPage
from diarytable_detection import DetectDiaryTable
from paddleocr import PaddleOCR
import numpy as np
import csv
from PyPDF2 import PdfReader
import re

ocr = PaddleOCR(lang="en")

class PDFImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diary to CSV Converter")
        self.root.geometry("600x400")

        # Initialize CustomTkinter theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # UI Elements
        self.label = ctk.CTkLabel(root, text="Diary to CSV converter", font=("Arial", 24))
        self.label.pack(pady=20)

        self.button_select_file = ctk.CTkButton(root, text="Select PDF File", command=self.select_pdf_file)
        self.button_select_file.pack(pady=10)

        self.button_select_folder = ctk.CTkButton(root, text="Select PDF Folder", command=self.select_pdf_folder)
        self.button_select_folder.pack(pady=10)

        self.process_button = ctk.CTkButton(root, text="Process PDFs", command=self.process_pdfs)
        self.process_button.pack(pady=20)

        self.status_label = ctk.CTkLabel(root, text="", font=("Arial", 14))
        self.status_label.pack(pady=10)

        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(root, orientation="horizontal", mode="determinate")
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

        self.pdf_files = []

    def select_pdf_file(self):
        # Select single PDF file
        file_path = filedialog.askopenfilename(title="Select PDF file", filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            self.pdf_files = [file_path]
            self.status_label.configure(text=f"Selected: {os.path.basename(file_path)}")

    def select_pdf_folder(self):
        # Select folder containing PDF files
        folder_path = filedialog.askdirectory(title="Select Folder")
        if folder_path:
            self.pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
            self.status_label.configure(text=f"Selected {len(self.pdf_files)} PDF files")

    def process_pdfs(self):
        if not self.pdf_files:
            messagebox.showerror("Error", "Please select a PDF file or folder first")
            return

        # Disable buttons and show loading cursor during processing
        self.root.config(cursor="wait")
        self.process_button.configure(state="disabled")
        self.button_select_file.configure(state="disabled")
        self.button_select_folder.configure(state="disabled")

        num_files = len(self.pdf_files)
        for idx, pdf_file in enumerate(self.pdf_files):
            self.process_pdf(pdf_file)
            # Update progress bar
            progress = (idx + 1) / num_files
            self.progress_bar.set(progress)
            self.status_label.configure(text=f"Processed {idx + 1} of {num_files} files")
            self.root.update_idletasks()  # Ensure the GUI remains updated during processing

        # Enable buttons and reset cursor after processing
        self.root.config(cursor="")
        self.process_button.configure(state="normal")
        self.button_select_file.configure(state="normal")
        self.button_select_folder.configure(state="normal")
        
        messagebox.showinfo("Success", "PDF Processing completed!")
        self.status_label.configure(text="Processing Completed")

    def process_pdf(self, pdf_file):
        pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]
        output_folder = os.path.join(os.path.dirname(pdf_file), pdf_name + "_processed")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        total_pages = len(PdfReader(pdf_file).pages)
        print(f"Total number of pages: {total_pages}")

        for page_number in range(456, total_pages + 1):
            try:
                # Convert one page at a time
                images = convert_from_path(pdf_file, first_page=page_number, last_page=page_number)
                
                for image in images:
                    open_cv_image = np.array(image)
                    # Convert RGB to BGR
                    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
                    self.process_diary_page(open_cv_image, pdf_name)
                    # Update status for each page
                    self.status_label.configure(text=f"Processed {pdf_file} - Page {page_number}")
                    self.root.update_idletasks()  # Keep UI responsive

            except Exception as e:
                print(f"Error processing page {page_number}: {str(e)}")
                break  # Stop processing if an error occurs (e.g., no more pages)

    def process_diary_page(self, image, pdf_name):
        image_height, image_width, _ = image.shape
        if DetectDiagyPage(image):
            detection_values = DetectDiaryTable(image)
            page_info = []

            location = ""
            date = ""

            prev_left = int(image_width / 20)
            prev_right = int(image_width * 19 / 20)

            for detection in detection_values:
                class_id = detection['class_id']
                class_name = detection['class_name']
                confidence = detection['confidence']
                box = detection['box']
                scale = detection['scale']

                # Calculate bounding box coordinates
                left, top, right, bottom = (
                    round(box[0] * scale),
                    round(box[1] * scale),
                    round((box[0] + box[2]) * scale),
                    round((box[1] + box[3]) * scale)
                )
                left = max(left, 0)
                top = max(top, 0)
                right = min(right, image.shape[1] - 1)
                bottom = min(bottom, image.shape[0] - 1)

                if class_name == "row" and right - left < int(image_width * 5 / 7):
                    left = prev_left
                    right = prev_right
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                else:
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

                # Crop the image to the region specified by the bounding box
                cropped_image = image[top:bottom, left:right]

                # Perform OCR on the cropped image
                result = ocr.ocr(cropped_image, cls=True)

                first_time, second_time, phone = "", "", ""
                activity_text = ""

                if class_name == 'row':
                    h, w = cropped_image.shape[:2]
                    timephone_image = cropped_image[0:h, 0:int(w / 4)]
                    time_result = ocr.ocr(timephone_image, cls=True)
                    time_text = ""

                    prev_left = left
                    prev_right = right
                    for line in time_result:
                        if line:
                            for word_info in line:
                                if len(word_info) < 2:
                                    continue
                                box, (text, score) = word_info
                                left = int(box[0][0])
                                top = int(box[0][1])
                                right = int(box[2][0])
                                bottom = int(box[2][1])
                                time_text += " " + text

                    pattern = r'(\d{1,2}:\d{2})\s*(\d{1,2}:\d{2})?\s*([RP])?'
                    match = re.search(pattern, time_text)
                    if match:
                        first_time = match.group(1)
                        second_time = match.group(2) if match.group(2) else ""
                        phone = match.group(3) if match.group(3) else ""

                    activity_image = cropped_image[0:h, int(w / 4):w]
                    activity_result = ocr.ocr(activity_image, cls=True)
                    for line in activity_result:
                        if line:
                            for word_info in line:
                                if len(word_info) < 2:
                                    continue
                                box, (text, score) = word_info
                                activity_text += " " + text

                elif class_name == 'location':
                    location_text = ""
                    for line in result:
                        for word_info in line:
                            if len(word_info) < 2:
                                continue
                            box, (text, score) = word_info
                            location_text += " " + text
                    location = location_text

                elif class_name == 'date':
                    date_text = ""
                    for line in result:
                        for word_info in line:
                            if len(word_info) < 2:
                                continue
                            box, (text, score) = word_info
                            date_text += " " + text
                    date = date_text

                row_info = {'Filename': f"{pdf_name}.pdf", 'Location': location, 'Date': date, 'FromTime': first_time, 'ToTime': second_time, 'Phone': phone, 'Activity': activity_text}
                page_info.append(row_info)

            csv_filename = f"{pdf_name}.csv"
            with open(csv_filename, mode='w', newline='') as csv_file:
                fieldnames = ['Filename', 'Location', 'Date', 'FromTime', 'ToTime', 'Phone', 'Activity']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for info in page_info:
                    writer.writerow(info)

            print(f"CSV file saved: {csv_filename}")

if __name__ == "__main__":
    root = ctk.CTk()
    app = PDFImageProcessorApp(root)
    root.mainloop()
