import os
import csv
from tqdm import tqdm
# Path to the folder containing text files
image_folder = r"C:\Users\Asus\Documents\Surrey\Research Project\CodeAnalysis\Caption Generation\a-PyTorch-Tutorial-to-Image-Captioning-master\a-PyTorch-Tutorial-to-Image-Captioning-master\fscoco\FSCOCO_1\FscocoCaptions"

# List to store the content of each text file
file_contents = []
image_ref_fold_no = list(os.listdir(image_folder))
for image_folder_no in tqdm(image_ref_fold_no):
    # print(f"Loading Action: {action}")
    
    image_path = os.path.join(image_folder, image_folder_no)
    for filename in os.listdir(image_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(image_folder, image_folder_no, filename)
            print(file_path)
            try:
                with open(file_path) as f:
                    captions = f.read()
                file_contents.append(captions)
            except Exception as e:
                print(f"Error processing file '{file_path}': {str(e)}")


csv_file_path =os.path.join(folder_path, "output.csv")

with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    for content in file_contents:
        csv_writer.writerow([content])

print("Content from text files has been written to the CSV file.")