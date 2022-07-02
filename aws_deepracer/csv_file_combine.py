import os
import csv


if __name__ == '__main__':

    input_file_dir = r'C:\Users\asus\Desktop\2022 aws\autobus-v4-training_job_inn-9A0nR9uvRoVacnZb1A_logs\7a442136-7e8b-43bc-82a0-7209e443b3da\sim-trace\training\training-simtrace'
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
