import argparse

import os
import numpy as np
import random
import errno
import glob
import xml.etree.cElementTree as etree
import parsers
import common
import itertools

def _write_to_file(filename, centroids):
     with open(filename, 'r+') as f:
        f.truncate()        
        for centroid in centroids:
            f.write("{0},{1} ".format(centroid[0], centroid[1]))

def _save_to_file(centroids, filename):
    filename = os.path.abspath(filename)

    try:
        file_handle = os.open(filename, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as e:
        if e.errno != errno.EEXIST:           
            raise
    
    _write_to_file(filename, centroids)

    print ("Saved to {0}".format(filename))

def _cluster_bounding_boxes(bounding_boxes, num_anchors=5):
    # Take random tuples from bounding_boxes to initialize default centroids
    indices = [ random.randrange(bounding_boxes.shape[0]) for i in range(num_anchors)]
    centroids = bounding_boxes[indices]
    
    centroids = _cluster(bounding_boxes, centroids)
    print (centroids)
    
    return centroids

def calculate(folder, num_anchors=5, out_file=None):
    pattern = '{0}/*{1}'.format(folder, '.xml')
    filenames = glob.glob(pattern)    

    if len(filenames) <= 0:
        print ("Folder {0} is empty".format(folder))
        return

    bounding_boxes = []
    for fl in filenames:
        info = parsers.parse_from_pascal_voc_format(fl)
        if len(info) < 2:
            continue
        for bb in info[1]:
            xn, yn, xx, yx = bb
            bounding_boxes.append([abs(xx-xn), abs(yx-yn)])

    bounding_boxes = np.array(bounding_boxes, dtype=np.int32)
    
    print ("Found {0} bounding boxes", len(bounding_boxes))

    centroids = _cluster_bounding_boxes(bounding_boxes, num_anchors)
    if out_file is not None:
        _save_to_file(centroids, out_file)
    return centroids
    
def _cluster(bounding_boxes, centroids, eps=0.05, iterations=100):
    """
        Cluster existing bounding boxes to N classes.
        Based on K-Means clustering algorithm
        Args: 
            bounding_boxes: list of tuples (w, h) for each bounding box
            centroids: list of random defined centroids based on existing bounding boxes
                       represented as tuples (w, h)
            eps: float that indicates stop factor of clustering (converge threshold)
            iterations: int (max number of iterations)
        
        Returns: 
            centroids : average tuple (w, h) for each cluster
    """
    
    distances, previous_distances = [], [] 
    iteration, difference = 0, 1e5
    centroids_count, centroid_size = centroids.shape

    while True:
        iteration+=1
        
        # Calculate distances (1 - IOU) between each bounding box and centroid
        # Based on YOLO v2 data preparation procedure
        distances = []          
        for i in range(bounding_boxes.shape[0]):
            distances.append((1 - _iou(bounding_boxes[i],centroids)))
        distances = np.array(distances)
        
        # Calculate difference between new/old (converge check)
        if len(previous_distances) > 0:
            difference = np.sum(np.abs(distances-previous_distances))
        
        print ("Iteration {0} : difference = {1}".format(iteration, difference))
        
        # Exit loop if centroids converged or reached max number of iterations 
        if difference < eps or iteration > iterations:
            print ("Iterations took = %d"%(iteration))
            return centroids

        # Assign data points to closest centroids
        # For each element in the dataset, chose the closest centroid. 
        closest_centroids = np.argmin(distances,axis=1)

        # Calculate new centroids (Each centroid is the geometric mean of the points that closest to it)
        centroid_sums=np.zeros((centroids_count,centroid_size),np.float)
        for i in range(closest_centroids.shape[0]):
            centroid_sums[closest_centroids[i]]+=bounding_boxes[i]
        
        for j in range(centroids_count):   
            sm = np.sum(closest_centroids==j)
            if sm != 0:         
                centroids[j] = centroid_sums[j]/sm
        
        previous_distances = distances.copy()
    return centroids


def _iou(x,centroids):
    """
        Calculates intersection over union between (w,h) and every centroids (cw, ch)
        Args: 
            x: tuple (width, height) of bounding box
            centroids: list of tuples (width, height) that represents average (w, h) of each cluster

        Returns: 
            Array of iou's 
    """

    dists = []
    w, h = x #bounding box size   
    for centroid in centroids:
        cw,ch = centroid #average bounding box size
        if cw>=w and ch>=h: # if centroid bigger in all dimensions   
            dist = w*h/(cw*ch)
        elif cw>=w and ch<=h: #if centroid bigger only in w
            dist = w*ch/(w*h + ch*(cw-w)) # add external part of initial box (c_h*(c_w-w))
        elif cw<=w and ch>=h: #if centroid bigger only in h
            dist = cw*h/(w*h + cw*(ch-h)) # add external part of initial box (c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            dist = (cw*ch)/(w*h)
        dists.append(dist)
    return np.array(dists)

def normalize(anchors, image_size, map_size):    
    rh, rw = [ float(a) / b  for a, b in zip(map_size, image_size)]

    normalized = []
    for anchor in anchors:
        w, h = rw * float(anchor[0]), rh * float(anchor[1])
        normalized.append(w)
        normalized.append(h)
    return normalized

def draw_anchors(anchors ):
    
    if len(anchors) < 1:
        print ("There is no anchors to show")
        return

    w, h = (np.max(anchors, axis=0) + 50)

    blank_image = np.zeros((h, w, 3), dtype=np.uint8)
    colors = [(255,0,0),(255,255,0),(0,255,0),(0,0,255),(0,255,255),(55,0,0),(255,55,0),(0,55,0),(0,0,25),(0,255,55)]
    palette = itertools.cycle(colors)

    dt = 5
    for anchor in anchors:
        dt += 5
        common.draw_bounding_box(blank_image, (dt, dt, anchor[0], anchor[1]), False, color=next(palette))
    
    return blank_image
   
