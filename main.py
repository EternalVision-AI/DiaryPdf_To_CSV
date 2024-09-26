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
from spellchecker import SpellChecker
import re
import csv
ocr = PaddleOCR(
    lang="en",  # Specify the language

)



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

        for pdf_file in self.pdf_files:
            self.process_pdf(pdf_file)
        
        messagebox.showinfo("Success", "PDF Processing completed!")
        self.status_label.configure(text="Processing Completed")

    def process_pdf(self, pdf_file):
        # Convert PDF to images and process each page
        pages = convert_from_path(pdf_file)
        pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]
        output_folder = os.path.join(os.path.dirname(pdf_file), pdf_name + "_processed")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for page_num, page_image in enumerate(pages, start=1):
            image_path = os.path.join(output_folder, f"page_{page_num}.png")

            # Save the page as an image (PIL image -> OpenCV format)
            page_image_cv = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)

            # Process the image using OpenCV
            processed_image = self.ProcessDiaryPage(page_image_cv, pdf_name)

            # Save processed image
            # cv2.imwrite(image_path, processed_image)

            self.status_label.configure(text=f"Processed {pdf_file} - Page {page_num}")
    # Function to correct spelling

    def ProcessDiaryPage(self, image, pdf_name):
        image_height, image_width, _= image.shape
        if DetectDiagyPage(image):
          detection_values = DetectDiaryTable(image)
          extracted_data = {
              "rows": []  # List to hold info for each detected row
          }
          

          page_info = []
          # Iterate over the detections and extract values based on class_name
          location = ""
          date = ""
          
          #temp variables
          prev_left = int(image_width/20)
          prev_right = int(image_width*19/20)
          for detection in detection_values:
              class_id = detection['class_id']
              class_name = detection['class_name']
              confidence = detection['confidence']
              box = detection['box']
              scale = detection['scale']
              
              row_info = None

              # Calculate bounding box coordinates
              left, top, right, bottom = round(box[0] * scale), round(box[1] * scale), round(
                  (box[0] + box[2]) * scale), round((box[1] + box[3]) * scale)
              if left<0: left = 0
              if top<0: top = 0
              if right>image.shape[1]: right = image.shape[1] - 1
              if bottom>image.shape[0]: bottom = image.shape[0] - 1
              
              if class_name == "row" and right - left < int(image_width*5/7):
                  left = prev_left
                  right = prev_right
                  cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
              else:
                  cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
                  
              # Crop the image to the region specified by the bounding box
              cropped_image = image[top:bottom, left:right]  # Crop using NumPy slicing
              

              origianl_left = left
              origianl_top = top
              origianl_right = right
              origianl_bottom = bottom
              result = None
              # Perform OCR on the cropped image
              if class_name != "row":
                result = ocr.ocr(cropped_image, cls=True)
              # Initialize a dictionary for the current row if the class is 'row'
              
              
              first_time, second_time, phone = "", "", ""
              activity_text = ""
              
              
              if class_name == 'row':
                  h, w = cropped_image.shape[:2]
                  timephone_image = cropped_image[0:h, 0:int(w/4)]  # Crop using NumPy slicing
                  result = ocr.ocr(timephone_image, cls=True)     
                  time_text = ""
                  
                  prev_left = left
                  prev_right = right
                  for line in result:
                    if line:
                      for word_info in line:
                          if len(word_info) < 2:
                              continue  # Skip incomplete word info
                          box, (text, score) = word_info
                          
                          # Draw the rectangle around the detected text
                          left = int(box[0][0])
                          top = int(box[0][1])
                          right = int(box[2][0])
                          bottom = int(box[2][1])
                          # corrected_text = correct_spelling(text)
                          # Draw rectangle on the image
                          cv2.rectangle(image, (origianl_left+left, origianl_top+top), (origianl_left+right, origianl_top+bottom), (0, 255, 0), 2)  # Green rectangle with thickness 2
                          
                          time_text = time_text + " " + text
                  # def extract_time_and_R(input_string):
                  # Regular expression to match the time format and "R"
                  print("time_text", time_text)
                  pattern = r'(\d{1,2}:\d{2})\s*(\d{1,2}:\d{2})?\s*([RP])?'
                  match = re.search(pattern, time_text)
                  if match:
                      first_time = match.group(1)
                      second_time = match.group(2) if match.group(2) else ""
                      phone = match.group(3) if match.group(3) else ""
                  print(f"First Time: '{first_time}', Second Time: '{second_time}', Phone: '{phone}'")

                  
                  activity_image = cropped_image[0:h, int(w/4):w]  # Crop using NumPy slicing
                  result = ocr.ocr(activity_image, cls=True)     
                  for line in result:
                    if line:
                      for word_info in line:
                          if len(word_info) < 2:
                              continue  # Skip incomplete word info
                          box, (text, score) = word_info
                          
                          # Draw the rectangle around the detected text
                          left = int(box[0][0])
                          top = int(box[0][1])
                          right = int(box[2][0])
                          bottom = int(box[2][1])
                          # corrected_text = correct_spelling(text)
                          # Draw rectangle on the image
                          cv2.rectangle(image, (origianl_left+left + int(w/4), origianl_top+top), (origianl_left+right + int(w/4), origianl_top+bottom), (255, 0, 0), 2)  # Green rectangle with thickness 2
                          
                          activity_text = activity_text + " " + text
                  print(f"activity: {activity_text}")

              # Add the detected 'location' and 'date' to the current row
              elif class_name == 'location':
                  location_text = ""
                  for line in result:
                    for word_info in line:
                        if len(word_info) < 2:
                            continue  # Skip incomplete word info
                        box, (text, score) = word_info
                        # corrected_text = correct_spelling(text)
                        location_text = location_text + " " + text
                  location = location_text
                  print(f"location: {location_text}")
              elif class_name == 'date':
                  date_text = ""
                  for line in result:
                    for word_info in line:
                        if len(word_info) < 2:
                            continue  # Skip incomplete word info
                        box, (text, score) = word_info
                        # corrected_text = correct_spelling(text)
                        date_text = date_text + " " + text
                  date = date_text
                  print(f"date: {date_text}")

              row_info = {'Filename': f"{pdf_name}.pdf", 'Location': location, 'Date': date, 'FromTime': first_time, 'ToTime': second_time, 'Phone': phone, 'Activity': activity_text}
              page_info.append(row_info)
          # Specify the CSV file to write to
          csv_filename = f"{pdf_name}.csv"

          # Append the page_info data to the CSV
          with open(csv_filename, mode='a', newline='') as file:
              writer = csv.DictWriter(file, fieldnames=['Filename', 'Location', 'Date', 'FromTime', 'ToTime', 'Phone', 'Activity'])
              def is_file_empty(filename):
                return not os.path.exists(filename) or os.path.getsize(filename) == 0
              # Write the header only if the file is new or empty
              if is_file_empty(csv_filename):
                  writer.writeheader()
              
              # Write each row of data
              for row in page_info:
                if row['Activity']:
                  writer.writerow(row)

          print(f"CSV file '{csv_filename}' created successfully.")
          height, width, _ = image.shape
          cv2.imshow("Diarypage_Detection", cv2.resize(image, (int(width/2), int(height/2))))
          cv2.waitKey(0)
        

        


if __name__ == "__main__":
    root = ctk.CTk()
    app = PDFImageProcessorApp(root)
    root.mainloop()
