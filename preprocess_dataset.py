import os

def clean_up_files(filepath, dir):
    files = []
    lines = []
    with open(filepath, 'r') as fp:
        files = fp.readlines()
    lines = files.copy()
    image_dir = f'{dir}/Image'
    c = 0
    for line in files:
        f = line.split(' ')[0]
        pt = f'{image_dir}/{f}'
        if not(os.path.exists(pt)):
            c+=1
            lines.remove(line)
            
    with open(filepath, 'w') as fp:
        fp.writelines(lines)
    print('No of lines removed: ', c) 
    

cam_test_files = './data/Info/CAM_test.txt'
cam_train_files = './data/Info/CAM_train.txt'
noncam_test_files = './data/Info/NonCAM_test.txt'
noncam_train_files = './data/Info/NonCAM_train.txt'
train_dir = './data/Train'
test_dir = './data/Test'

clean_up_files(cam_train_files, train_dir)
clean_up_files(noncam_train_files, train_dir)
clean_up_files(cam_test_files, test_dir)
clean_up_files(noncam_test_files, test_dir)
