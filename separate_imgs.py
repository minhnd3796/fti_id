from shutil import move
from os import listdir, mkdir
from os.path import join, exists, isfile

input_dir = 'dob'
total_file = listdir(input_dir)

file_index = 0
max_item_per_dir = 500
sub_dir_index = 0
for file in total_file:
    if isfile(join(input_dir, file)):
        output_dir = join(input_dir, str(sub_dir_index))
        if file_index % max_item_per_dir == 0:
            sub_dir_index += 1
            output_dir = join(input_dir, str(sub_dir_index))
            if not exists(output_dir):
                mkdir(output_dir)
        print('Moving', join(output_dir, file))
        move(join(input_dir, file), join(output_dir, file))
        file_index += 1
