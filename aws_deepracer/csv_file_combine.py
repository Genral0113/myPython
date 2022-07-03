import os
import csv


if __name__ == '__main__':

    input_file_dir = r'C:\Users\asus\Desktop\2022 aws\model-v7-training_job_qMARHs5_T0WxP8RmuQ837Q_logs\74dff1a2-b748-4975-aea9-3c9de4f79e35\sim-trace\training\training-simtrace'
    output_file_name = r'\all-iterations.csv'

    skip_header = False
    for i in range(len(os.listdir(input_file_dir))):
        file_name = str(i) + '-iteration.csv'
        if os.path.isfile(os.path.join(input_file_dir, file_name)):
            if file_name.split('.')[1] == 'csv':
                input_file_full_name = os.path.join(input_file_dir, file_name)
                output_file_full_path = input_file_dir + output_file_name
                output_file = open(output_file_full_path, 'a')
                with open(input_file_full_name) as fil:
                    row = fil.readline()
                    if not skip_header:
                        output_file.write(row)
                        skip_header = True
                    for row in fil.readlines():
                        output_file.write(row)
