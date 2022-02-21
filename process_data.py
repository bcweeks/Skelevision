import os
import cv2
import json
import glob
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

ANNOTATION_FILE = 'annotations/oct_2021_annotation.json'
PROCESSED_IMG_DIR = 'data/processed_images/'


def convert_image(input_addr):
    '''
    Helper function to convert images from TIFF to JPG
    '''
    skeleton_id = input_addr[input_addr.find('UMMZ_')+5:input_addr.find('/skeleton')]
    rgb = cv2.imread(input_addr)
    rgb = cv2.resize(rgb, (5472, 3648))
    cv2.imwrite(PROCESSED_IMG_DIR + skeleton_id + '.jpg', rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def process_data(root_dir='data/Training_Images/'):
    '''
    Main function to convert all images to JPG and validate annotation file
    '''
    
    #! CHECK ROOT DIRECTORY
    if not os.path.isdir(root_dir):
        print(f'ERROR: root data directory missing {root_dir}')
        print('please check readme and downlaod training data to data/')
        return
    else:
        print('found root data directory')
    
    
    #! MAKE DIRECTORY FOR PROCESSED IMAGES
    if not os.path.isdir(PROCESSED_IMG_DIR):
        os.makedirs(PROCESSED_IMG_DIR)
        
    
    #! GET IMAGE ADDRESS
    skeleton_list = glob.glob(root_dir + 'UMMZ*')
    print(f'found {len(skeleton_list)} specimens')
    image_list = glob.glob(root_dir + 'UMMZ*/*.tiff')
    print(f'found {len(image_list)} images')
    
    
    #! CONVERT IMAGES ONLY IF IT DOESN'T EXIST
    existing = set()
    existing_images = glob.glob(PROCESSED_IMG_DIR + '*.jpg')
    for images in existing_images:
        existing.add(images.replace(PROCESSED_IMG_DIR,'').replace('.jpg',''))
    print(f'found {len(existing)} processed images')
    
    
    #! MAKE LIST OF IMAGES TO PROCESS
    process_list = []
    for images in image_list:
        skeleton_id = images[images.find('UMMZ_')+5:images.find('/skeleton')]
        if skeleton_id not in existing:
            process_list.append(images)
    
    
    #! CONVERT IAMGES TIFF TO JPG
    print(f'processing {len(process_list)} images')
    process_map(convert_image, process_list, max_workers=4, chunksize=1)
    
    
    #! UPDATE PROCESSED IMAGES LIST
    existing = set()
    existing_images = glob.glob(PROCESSED_IMG_DIR + '*.jpg')
    for images in existing_images:
        existing.add(images.replace(PROCESSED_IMG_DIR,'').replace('.jpg',''))
    
    
    #! READ ANNOTATION FILE
    if not os.path.isfile(ANNOTATION_FILE):
        print(f'ERROR: annotation file not found ({ANNOTATION_FILE})')
        return
    with open(ANNOTATION_FILE, 'r') as fp:
        data = json.load(fp)
    
    
    #! VALIDATE ALL IMAGES ARE PROCESSED
    print(f'loaded {len(data["images"])} annoated images')
    all_found = True
    for item in data['images']:
        skeleton_id = item['file_name'].replace('skeleton-','').replace('.jpg','')
        if skeleton_id not in existing:
            print('image not found', skeleton_id)
            all_found = False
        
    if all_found:
        print('SUCCESS: all images processed')
    else:
        print('ERROR: some images not found')
    

if __name__ == '__main__':
    process_data()