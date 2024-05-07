
def get_labels(filepath, label_arr, label_dict):
    li = len(label_arr) - 1
    with open(filepath, 'r') as fp:
        data_paths = fp.readlines()
        for line in data_paths:
            fn = line.split(' ')[0]
            info = fn.split('-')
            if info[-2].lower() not in label_arr:
                li += 1
                label_dict.append({
                    'name': info[-2].lower(),
                    'label_index': li
                })
                label_arr.append(info[-2].lower())
    return label_arr, label_dict


test_path = 'source-data/Info/CAM_test.txt'
train_path = 'source-data/Info/CAM_train.txt'

label_arr = []
label_dict = []

label_arr.append('noncam')
label_dict.append({
    'name': 'noncam',
    'label_index': 0
})
label_arr, label_dict = get_labels(train_path, label_arr, label_dict)
label_arr, label_dict = get_labels(test_path, label_arr, label_dict)
print(label_arr)
print(label_arr.index('monkey'))
print(label_dict)