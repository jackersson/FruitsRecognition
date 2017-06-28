import sys
import os
import utils
import parsers
import shutil

def split(data):
    n = len(data)
    tn_s, tn_e = 0, int(n*.8)
    ts_s, ts_e   = tn_e, n
    print ("Images splitted to train({0}), test({1})".format(tn_e, (ts_e - ts_s)))
    return data[tn_s:tn_e], data[ts_s:ts_e]

def split_all(labels, folder='.', out_folder='.'):
    if len(labels) <= 0:
        print ("Labels count can't be 0")
        return
    
    for lbl in labels:
          _split_data(lbl, folder, out_folder)

def _save(files, folder='.', train=True, image=True):
    """
        Save files to folders: 
            train/test  / images/annotations
    """
    folder = os.path.abspath(folder)
    
    postfix = "images" if image else "annotations"
    ops     = "train" if train else "test"
    
    target_dir = os.path.join(folder, '{0}/{1}'.format(ops, postfix))
    
    utils.make_dir(target_dir)
    
    cnt = 0
    for fl in files:
        if os.path.exists(fl):
            shutil.copy(fl, target_dir)
            cnt += 1
        else:
            print ("Not exists {0}".format(fl))
    
    print ("Saved {0} files to {1}".format(cnt, target_dir))
    return target_dir

#Helpers
def _remove_without_image(files, folder='.'):    
    folder = os.path.abspath(folder)
    if not os.path.exists(folder):
        print ("Image folder not exists")
        return []
    
    images = parsers.list_files(folder, '.jpg')
    if len(images) <= 0:
        print ("Images folder is empty")
        return []
        
    filtered = []
    for fl in files:
        filename = utils.get_filename(fl)
        image_file = os.path.join(folder, filename + '.jpg' )
        
        if os.path.exists(image_file):
            filtered.append(fl)
    return filtered   

def _fetch_images_from_annotations(annotations):    
    annotations = os.path.abspath(annotations)
    if not os.path.exists(annotations):       
        print ("Annotations folder not exists")
        return    

    files = parsers.list_files(annotations, '.xml')
    
    images = []
    for fl in files:
        res = parsers.parse_from_pascal_voc_format(fl)
        if res is not None and len(res) > 0:
            images.append(res[0])
        else:
            print ("Error parsing {0}".format(fl))
    
    return images   

def _prepend_image_path(filename, folder='', extension=''): 
    
    if extension != '':
        filename = filename + extension 
        
    if folder != '':
        filename = os.path.join(folder, filename)
    
    return filename

def _prepend_images_path(filenames, folder='', extension=''):
    result = []
    for fl in filenames:
        result.append(_prepend_image_path(fl, folder, extension))
    return result

def _split_data(class_name, folder='.', out_folder='.'):
    
    print ('Processing {0}...'.format(class_name))
    
    folder = os.path.abspath(folder)
    
    annotations   = os.path.join(folder, 'annotations')
    images_folder = os.path.join(folder, 'images/{0}'.format(class_name))
    
    files = parsers.list_files(os.path.join(annotations, class_name), '.xml')
    
    files = _remove_without_image(files, images_folder)
    if len(files) <= 0:
        print ('Annotations not correspond images for {0}'.format(class_name))
        return
    
    data = utils.randomize(files)
    train, test = split(data)
    
    out_folder = os.path.abspath(out_folder)
   
    ann_train_dir = _save(train, out_folder, image=False)
    ann_test_dir  = _save(test , out_folder, train=False, image=False)
    
    images = _fetch_images_from_annotations(ann_train_dir)
    images = _prepend_images_path(images, images_folder, '.jpg')    
    _save(images, out_folder)
    
    images = _fetch_images_from_annotations(ann_test_dir)
    images = _prepend_images_path(images, images_folder, '.jpg')    
    _save(images, out_folder, train=False)
   
    