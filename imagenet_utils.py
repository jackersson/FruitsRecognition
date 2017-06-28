import sys
import os
import time
import tarfile
import shutil
import cv2
import parsers
import random
import common
import utils


if sys.version_info >= (3,):
    import urllib.request as urllib2
    import urllib.parse as urlparse
else:
    import urllib2
    import urlparse
    import urllib

min_image_size = 10 * 1024


class ImageNetLoader(object):

    def __init__(self):        
        self.host = "http://www.image-net.org"
        self.images      = "{0}{1}".format(self.host, '/api/text/imagenet.synset.geturls?wnid=')
        self.mappings    = "{0}{1}".format(self.host, '/api/text/imagenet.synset.geturls.getmapping?wnid=')
        self.annotations = "{0}{1}".format(self.host, '/downloads/bbox/bbox/')
        self.names       = "{0}{1}".format(self.host, '/api/text/wordnet.synset.getwords?wnid=')

    def download(self, class_ids, folder='.', params={}):
        maps = {}
        for class_id in class_ids:
            print ("Downloading {0}".format(class_id))
            class_name = self._download_single(class_id, folder, params)
            maps[class_id] = class_name
        return maps
        
    def _download_single(self, class_id, folder='.', params={}):
        folder = os.path.abspath(folder)
        
        class_name = self.get_class_name(class_id)
        if class_name is None:
            class_name = class_id
        

        print ("Class name - {0}".format(class_name)) 
        
        if 'boxes' in params.keys() and params['boxes']: 
            annotations_dir = self.download_annotations(class_id, folder, class_name)
            if 'set_name' in params.keys() and params['set_name'] and class_name != class_id:
                print (annotations_dir,  class_name)
                files = parsers.list_files(os.path.join(annotations_dir, class_name), '.xml')
                parsers.set_object_name(files, class_name)

        if 'images' in params.keys() and params['images']:
            self.download_images(class_id, folder, class_name)

        return class_name    

    def get_class_name(self, class_id):
        url = self.names + class_id
        try:
            response = urllib2.urlopen(url)   
            name     = response.read().decode('utf-8')
            return name.rstrip()
        except (Exception) as error:
            print (error)
            print ('Fail to download ' + url)
        return None

    def download_annotations(self, class_id, folder='.', class_name=None, force=False):

        folder = utils.make_dir(os.path.abspath(os.path.join(folder, "annotations")))
        if force:
            utils.rmdir(os.path.join(folder, class_id))
            utils.rmdir(os.path.join(folder, class_name))

        temp_folder = utils.make_dir(os.path.join(folder, 'temp'))
        
        filename = str(class_id) + '.tar.gz'
        url = self.annotations + filename

        try:
            filename = os.path.join(temp_folder, filename)
            utils.download_file(url, filename)            
        
            current_dir = os.getcwd()
            os.chdir(temp_folder)

            utils.extract_tar_file(filename)
            print ("Extracted to folder {0}".format(folder))
            os.chdir(current_dir)

            shutil.move('{0}/Annotation/{1}'.format(temp_folder, class_id), folder)

            output = '{0}/{1}'.format(folder, class_id)
            if class_name is not None:
                name_output = '{0}/{1}'.format(folder, class_name)
                os.rename(output, name_output)
                output = name_output

            print ('Download box annotation from ' + url + ' to ' + output)
        except (Exception) as error:
            print (error)
            print ('Fail to download ' + url)  

        shutil.rmtree(temp_folder) 
        return folder

    def download_images(self, class_id, folder='.', class_name=None, with_maps=True, force=False):
        #TODO possible check on image content
        folder = utils.make_dir(os.path.abspath(os.path.join(folder, "images")))
        if force:
            utils.rmdir(os.path.join(folder, class_id))
            utils.rmdir(os.path.join(folder, class_name))

        output = os.path.join(folder, class_id)
        if class_name is not None:
            output = os.path.join(folder, class_name)
            
        utils.make_dir(output)

        urls = self.download_urls(class_id)
        if urls is None:
            return False
        
        mappings = None
        if with_maps:
            mappings = self.download_mappings(class_id)
            if mappings is None:
                with_maps = False
            
        cnt = 0
        for url in urls:
            if with_maps: 
                if url in mappings.keys():
                    filename = mappings[url] + '.jpg'
                else:               
                    continue
            else:
                filename = os.path.basename(url)

            filename = os.path.join(output, filename)
            if not os.path.exists(filename):
                if self._download_image(url, filename):
                    cnt += 1    
        print ("Done. {0} saved".format(cnt))
        return folder


    def download_urls(self, class_id ):
        url = self.images + class_id
        try:
            response = urllib2.urlopen(url)  
            contents = response.read().decode('utf-8').split('\n')
            image_url = []
            for each_line in contents:
                # Remove unnecessary char
                each_line = each_line.replace('\r', '').strip()
                if each_line:
                    image_url.append(each_line)
            return image_url

        except (Exception) as error:
            print ('Fail to download : ' + url)
            print (str(error))
        return None

    def download_mappings(self, class_id):
        url = self.mappings + class_id
        try:

            response = urllib2.urlopen(url)   
            contents = response.read().decode('utf-8').split('\n')     
            mappings = {}

            for each_line in contents:                
                # Remove unnecessary char
                each_line = each_line.replace('\r', '').strip()
                par = each_line.split(" ")
                if len(par) > 1:
                    filename, url = par[:2]
                    url = url.replace('jpeg', 'jpg')
                    mappings[url] = filename
               
            return mappings

        except (Exception) as error:
            print ('Fail to download : ' + url)
            print (str(error))
        return None   

    def _download_image(self, url, filename):
        try:
            utils.download_file(url, filename, min_size=min_image_size)
            return True
        except (Exception) as error:
            print ('Fail to download : ' + url)
            print (str(error))
            return False

def visualize_data(class_name, folder='.', filename=None):
    folder = os.path.abspath(folder)
    
    images_dir      = os.path.join(folder, 'images/{0}'.format(class_name))
    annotations_dir = os.path.join(folder, 'annotations/{0}'.format(class_name))
    
    if filename is not None:
        filename = os.path.join(annotations_dir, os.path.splitext(filename)[0] + '.xml')        
    else:
        annotations = parsers.list_files(annotations_dir, '.xml')
        if len(annotations) <= 0:
            print ("No annotations found")
            return None
        filename = annotations[0]
        
    info = parsers.parse_from_pascal_voc_format(filename)
    if len(info) <= 0:
        print ("Bounding box not found")
        return None    
        
    img = common.draw_annotations("{0}/{1}".format(images_dir,info[0] ), info[1])
    return img