import os
import csv

source = '/home/yuxiao/roberta'
files = os.listdir(source)
predictions = []
for file in files:
    if 'roberta_' in file and 'mnli' in file:
        lines = open(os.path.join(source, file), 'r').readlines()
        predictions.append(lines)

augmentation = {}
# strategy
# for i in range(len(predictions[0])):
#     flag = True
#     for j in range(len(predictions)):
#         if predictions[j][i] == predictions[0][i]:
#             continue
#         else:
#             flag = False
#     if flag:
#         augmentation[i] = predictions[0][i]

# strategy 1
# [sentence1, sentence2] -> [sentence2, sentence1]

# strategy 2
for i in range(len(predictions[0])):
    flag = True
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for j in range(len(predictions)):
        if predictions[j][i].strip() == '0':
            count_0 += 1
        elif predictions[j][i].strip() == '1':
            count_1 += 1
        else:
            count_2 += 1

    if count_0 > len(predictions) * 0.8:
        augmentation[i] = '0'
    elif count_1 > len(predictions) * 0.8:
        augmentation[i] = '1'
    elif count_2 > len(predictions) * 0.8:
        augmentation[i] = '2'


# strategy 3
for i in range(len(predictions[0])):
    flag = True
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for j in range(len(predictions)):
        if predictions[j][i].strip() == '0':
            count_0 += 1
        elif predictions[j][i].strip() == '1':
            count_1 += 1
        else:
            count_2 += 1

    if count_0 > len(predictions) * 0.5:
        augmentation[i] = '0'
    elif count_1 > len(predictions) * 0.5:
        augmentation[i] = '1'
    elif count_2 > len(predictions) * 0.5:
        augmentation[i] = '2'
train_data = csv.reader(open(os.path.join(source, 'train_update.csv'), 'r'))
val_data = csv.reader(open(os.path.join(source, 'test_phase_1_update.csv'), 'r'))
train_data_aug = csv.writer(open(os.path.join(source, 'train_update_aug4.csv'), 'w'))
for item in train_data:
    train_data_aug.writerow(item)
index = -1
for item in val_data:
    if index in augmentation.keys():
        temp = item + [augmentation[index].strip()]
        train_data_aug.writerow(item + [augmentation[index].strip()])
    index += 1