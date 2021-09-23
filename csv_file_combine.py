import os
import csv


if __name__ == '__main__':

    input_file_dir = os.path.join(os.path.dirname(__file__),  r'aws\training-simtrace\model9')
    output_file_full_path = os.path.join(input_file_dir,  r'all\all-iterations.csv')

    for file_name in os.listdir(input_file_dir):
        skip_header = False
        if os.path.isfile(os.path.join(input_file_dir, file_name)):
            if file_name.split('.')[1] == 'csv':
                input_file_full_name = os.path.join(input_file_dir, file_name)
                output_file = open(output_file_full_path, 'a')
                with open(input_file_full_name) as fil:
                    row = fil.readline()
                    if not skip_header:
                        output_file.write(row)
                        skip_header = True
                    for row in fil.readlines():
                        output_file.write(row)
