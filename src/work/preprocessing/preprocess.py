import os
import json
from tqdm import tqdm


def load_json_files(base_path):
    data = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    data.append(json_data)
    return data


def extract_relevant_data(json_data):
    extracted_data = []
    for entry in json_data:
        occupation = entry["dataSet"]["info"]["occupation"]
        gender = entry["dataSet"]["info"]["gender"]
        experience = entry["dataSet"]["info"]["experience"]
        question = entry["dataSet"]["question"]["raw"]["text"]
        answer = entry["dataSet"]["answer"]["raw"]["text"]
        extracted_data.append(f"Occupation: {occupation}\nGender: {gender}\nExperience: {experience}\nQuestion: {question}\nAnswer: {answer}\n")
    return extracted_data


def preprocess_data(base_path):
    json_data = load_json_files(base_path)
    extracted_data = extract_relevant_data(json_data)
    return extracted_data


def save_to_text_file(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(entry + "\n")


# Usage example
base_path = './채용면접데이터/Sample/02.라벨링데이터'
extracted_data = preprocess_data(base_path)
output_path = './preprocessed_data.txt'
save_to_text_file(extracted_data, output_path)
