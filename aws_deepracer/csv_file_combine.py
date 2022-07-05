import os
import csv


if __name__ == '__main__':

    input_file_dir = r'C:\Users\asus\Desktop\2022 aws\model-v7-training_job_ni5gD3LsRRaAUgDr7ZYmVg_logs\2e0f27ff-0f90-4ec1-91eb-ae764b38097f\sim-trace\training\training-simtrace'
    input_file_dir = r'C:\Users\asus\Desktop\2022 aws\Johnny4001-training_job_ZzKOgy1JROiqgYb9stilYQ_logs\811aa389-7e92-4ab7-99d7-cbdb51ee0a58\sim-trace\training\training-simtrace'
    input_file_dir = r'C:\Users\asus\Desktop\2022 aws\johnny4001-v2-training_job_zRVGgu1XThOtvo0RwWQ5wg_logs\e044c950-29c1-42ea-aa7e-ca6244245b5c\sim-trace\training\training-simtrace'
    input_file_dir = r'C:\Users\asus\Desktop\2022 aws\autobus-v8-training_job_8G2s1EelS9y6x_XxWolXYQ_logs\351e115e-1166-48b6-afdb-d5ce480d0756\sim-trace\training\training-simtrace'
    input_file_dir = r'C:\Users\asus\Desktop\2022 aws\autobus-final-training\b35a7d47-5f47-4fef-a4f9-c1be7e40f556\sim-trace\training\training-simtrace'
    input_file_dir = r'C:\Users\asus\Desktop\2022 aws\autobus-final-clone-training_job_6aekpJ1gTMO2SLjDsHvSgA_logs\0075f857-2f73-456c-8ba3-3c4bc5730e7c\sim-trace\training\training-simtrace'
    input_file_dir = r'C:\Users\asus\Desktop\2022 aws\autobus-final-v3-training_job_YHuE0dYIT1qQfkW35rMQEg_logs\c2bac156-961f-4c2d-bcd4-507e675c007e\sim-trace\training\training-simtrace'
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
