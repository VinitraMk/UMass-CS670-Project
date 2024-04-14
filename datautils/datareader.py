# this file is to read data folder
# and get information such as length, x and y

def read_data(type = 'Train', get_neg = False):
    image_path = f'data/{type}/Image'
    mask_path = f'data/{type}/GT_Object'
    edge_path = f'data/{type}/GT_Edge'
    pos_info_path = f'data/Info/CAM_{type.lower()}.txt'
    neg_info_path = f'data/Info/NonCAM_{type.lower()}.txt'
    #info_test_path = f'data/Info/CAM_test.txt'
    
    with open(pos_info_path, 'r') as fp:
        pos_data_info = fp.readlines()
    with open(neg_info_path, 'r') as fp:
        neg_data_info = fp.readlines()
        
    data_paths = []
    
    for line in pos_data_info:
        infos = line.split(" ")
        data_paths.append(
            {
                "image_path": f"{image_path}/{infos[0]}",
                "mask_path": (f"{mask_path}/{infos[0]}").replace("jpg", "png"),
                "camouflaged": infos[1]
            }
        )
    if get_neg:
        for line in neg_data_info:
            data_paths.append(
                {
                    "image_path": f"{image_path}/{infos[0]}",
                    "mask_path": (f"{mask_path}/{infos[0]}").replace("jpg","png"),
                    "camouflaged": infos[1]
                }
            )
    
    return data_paths
        