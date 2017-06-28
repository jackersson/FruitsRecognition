import xml.etree.cElementTree as etree
import numpy as np
import os
import glob

def list_files(folder, file_format='.jpg'):
    """
        List files in directory of specific format
        Args: 
            folder: dirname where to search files
            file_format: file extension to search
        
        Returns:
            filenames: list of strings
    """
    pattern = '{0}/*{1}'.format(folder, file_format)
    filenames = glob.glob(pattern)    
    return filenames

def parse_from_pascal_voc_format(filename):
    """
        Look to Pascal VOC 2007 XML format http://host.robots.ox.ac.uk/pascal/VOC/voc2007/guidelines.html
        Args: 
            filename: *.xml file with data according to Pascal VOC 2007 XML format

        Returns: 
            Bounding box: ints xmin, ymin, xmax, ymax - 
                          represents bounding box corners coordinates
    """
    in_file = open(filename)
    tree=etree.parse(in_file)
    root = tree.getroot()

    image_filename = root.find('filename').text

    bboxes = []
    for obj in root.iter('object'):
        current = list()              
        xmlbox = obj.find('bndbox')
        xn = int(float(xmlbox.find('xmin').text))
        xx = int(float(xmlbox.find('xmax').text))
        yn = int(float(xmlbox.find('ymin').text))
        yx = int(float(xmlbox.find('ymax').text))
        in_file.close()

        bboxes.append([xn, yn, xx, yx])        
    return [image_filename, bboxes]

def set_object_name(files, class_name):
    if len(files) <= 0:
        print ("Files can't be empty")
        return
    
    for fl in files:
        tree = etree.ElementTree(file=fl)
        root = tree.getroot()   

        image_filename = root.find('filename').text
        if not image_filename.endswith('.jpg'):
            root.find('filename').text = image_filename + '.jpg'
        for obj in root.iter('object'):       
            if class_name != '':     
                obj.find('name').text = class_name        

        tree=etree.ElementTree(root)
        tree.write(fl)     
       
    print ("Changed object names {0}".format(len(files)))


    
#TODO maybe remove 
def parse_from_json_darkflow_format(data):
    """
         Look to Darkflow prediction format https://github.com/thtrieu/darkflow
         Args: 
            data : dict with keys and values that corresponds to object detection prediction

         Returns: 
            Class confidence: float 
            Bounding box :  ints xmin, ymin, xmax, ymax - 
                            represents bounding box cornets coordinates

    """
    xmin = (data['topleft']['x'])
    ymin = (data['topleft']['y'])
        
    ymax = (data['bottomright']['y'])
    xmax = (data['bottomright']['x'])
    return data['confidence'], (xmin, ymin, xmax, ymax)