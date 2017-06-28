import os 
import shutil
import tarfile
import sys
import random

if sys.version_info >= (3,):
    import urllib.request as urllib2
    import urllib.parse as urlparse
else:
    import urllib2
    import urlparse
    import urllib

def get_filename(path):
    with_extension = os.path.basename(path)

    return os.path.splitext(with_extension)[0]

def randomize(data):   
    shuffled_index = list(range(len(data)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    data = [data[i] for i in shuffled_index]
    return data

def rmfile(filename):
    if os.path.exists(filename):
        os.remove(filename)

def rmdir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print ("Removed directory {0}".format(directory))
    return directory

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print ("Created directory {0}".format(directory))
    return directory

def extract_tar_file(filename):
    tar = tarfile.open(filename)
    tar.extractall()
    tar.close()

def download_file(url, filename, min_size=0):
    u = urllib2.urlopen(url)       
    
    with open(filename, 'wb') as f:
        meta = u.info()
        meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
        meta_length = meta_func("Content-Length")
        file_size = None
        if meta_length:
            file_size = int(meta_length[0])

        if file_size < min_size:
            print ("Skipped : {0} Bytes: {1} < {2}".format(url, file_size, min_size))
            f.close()
            rmfile(filename)
            return    

        print("Downloading: {0} Bytes: {1}".format(url, file_size))        

        file_size_dl = 0
        block_sz = 8192
        while True:
            buf = u.read(block_sz)
            if not buf:
                break

            file_size_dl += len(buf)
            f.write(buf)

            status = "{0:16}".format(file_size_dl)
            if file_size:
                status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
            status += chr(13)