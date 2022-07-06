import os
import csv

input_file_dir = r'C:\Users\asus\Desktop\2022 aws\v11-training_job_alMAWP4CTcSqJNvjzjGtLw_logs\04ef2818-b7fc-453b-b5d8-2818bf46a3db\sim-trace\training\training-simtrace'
output_file_name = r'\all-iterations.csv'

if __name__ == '__main__':
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
