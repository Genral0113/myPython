import os

home_path='M:\\'
for fil in os.listdir(home_path):
    if os.path.isfile(home_path+'\\'+fil):
        if fil.split('.')[0][-3:] == '(1)':
            os.remove(home_path+'\\'+fil)