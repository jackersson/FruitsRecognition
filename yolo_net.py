from darkflow.net.build import TFNet
import cv2
import os

import common
import numpy as np

import json


class YoloNet(object):
    
    def __init__(self, options):
        self.tfnet = TFNet(options)
        with open(options['metaLoad'], 'r') as fp:
            meta = json.load(fp)
            self.labels    = meta['labels']        
            self.threshold = meta['thresh']
            self.colors    = meta['colors']
    
        
    def detect(self, image):
        input_image = self._resize_input(image)
        this_inp = np.expand_dims(input_image, 0)
        feed_dict = {self.tfnet.inp : this_inp}

        res = self.tfnet.sess.run(self.tfnet.out, feed_dict)[0]
        boxes = self.tfnet.framework.findboxes(res)
        
        h, w = image.shape[:2]
        clean_boxes = []
        bx = []
        for box in boxes:
            tmpBox = self._process_box(box, h, w, self.threshold)
            if tmpBox is None:
                continue
            left, right, top, bot, class_name, class_indx, confidence = tmpBox
           
            clean_boxes.append([[class_indx, class_name, confidence], [left, right, top, bot]])
            
        #clean_boxes = non_intersect_boxes(clean_boxes)
        return clean_boxes    
    
            
    def draw_detections(self, image, boxes):
        if (len(boxes)) <= 0:
            return image
        h, w = image.shape[:2]
    
        thick = int((h + w) // 300)
        
        for box in boxes:
            class_indx, class_name, confidence = box[0]
            left, right, top, bot = box[1]        
    
            cv2.rectangle(image, (left, top), (right, bot), self.colors[class_indx], thick)
            cv2.putText  (image, class_name, (left, top - 12), 0, 1e-3 * h, self.colors[class_indx], thick//2)
        return image
           
        
    def _resize_input(self, im):
        h, w, c = self.tfnet.meta['inp_size']
        imsz = cv2.resize(im, (w, h))
        imsz = imsz / 255.
        imsz = imsz[:,:,::-1]
        return imsz

    def _process_box(self, b, h, w, threshold):
        max_indx = np.argmax(b.probs)
        max_prob = b.probs[max_indx]
        label = self.labels[max_indx]
        if max_prob > threshold:
            left  = int ((b.x - b.w/2.) * w)
            right = int ((b.x + b.w/2.) * w)
            top   = int ((b.y - b.h/2.) * h)
            bot   = int ((b.y + b.h/2.) * h)
            if left  < 0    :  left = 0
            if right > w - 1: right = w - 1
            if top   < 0    :   top = 0
            if bot   > h - 1:   bot = h - 1
            class_name = '{}'.format(label)
            return (left, right, top, bot, class_name, max_indx, max_prob)
        return None

def create_net(model_name, folder, options):
    options['pbLoad']   = os.path.join(os.path.abspath(folder), model_name + ".pb")
    options['metaLoad'] = os.path.join(os.path.abspath(folder), model_name + ".meta")
    return YoloNet(options)
