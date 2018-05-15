from pandas import read_csv, DataFrame
from ntpath import basename
from os import listdir
from numpy import array
from os import mkdir
from os.path import join, exists
from shutil import copy

table = read_csv('table.csv')

id_number = table.iloc[:, 1].values
filename = table.iloc[:, 4].values

for i in range(len(filename)):
    filename[i] = basename(filename[i])[:-4]

file_from_dir = listdir('id_good')
for i in range(len(file_from_dir)):
    file_from_dir[i] = file_from_dir[i][16:]
    file_from_dir[i] = file_from_dir[i][:-4]

available_pos_from_dir = []
available_pos_from_table = []
for i in range(len(filename)):
    if filename[i] in file_from_dir:
        pos_from_dir = file_from_dir.index(filename[i])
        available_pos_from_dir.append(pos_from_dir)
        available_pos_from_table.append(i)
id_labels_list = id_number[array(available_pos_from_table)].tolist()
filename_list = array(listdir('id_good'))[array(available_pos_from_dir)].tolist()

for i in range(len(id_labels_list)):
    id_labels_list[i] = str(id_labels_list[i])

new_csv_dict = {'id_label': id_labels_list, 'filename': filename_list}
df = DataFrame(data=new_csv_dict)
df.to_csv('labelled_images.csv', index_label='index')

if not exists('id_good_labelled'):
    mkdir('id_good_labelled')
for file in filename_list:
    copy(join('id_good', file), join('id_good_labelled', file))
