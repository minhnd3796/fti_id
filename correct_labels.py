from pandas import read_csv, DataFrame

dataset = read_csv('labelled_images.csv')
label = dataset.iloc[:, 2].values.tolist()
filename_list = dataset.iloc[:, 1].values.tolist()
print(len(label))

count_less_than_9 = 0
count_10 = 0
count_11 = 0

# for i in range(len(label)):
#     label[i] = str(label[i])
#     print(type(label[i]))

for i in range(len(label)):
    label[i] = str(label[i])
    if len(label[i]) == 10:
        label[i] = '00' +  label[i]
        count_10 += 1
    elif len(label[i]) == 10:
        label[i] = '0' +  label[i]
        count_11 += 1
    elif len(label[i]) == 8:
        label[i] = '0' +  label[i]
        count_less_than_9 += 1

print(count_less_than_9)
print('10:', count_10)
print('11:', count_11)

new_csv_dict = {'id_label': label, 'filename': filename_list}
df = DataFrame(data=new_csv_dict)
df.to_csv('corrected_labelled_images.csv', index_label='index')