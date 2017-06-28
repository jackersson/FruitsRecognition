import os 
import cv2


def load_image(filename, flags=-1): 
    if not os.path.exists(filename):
        print ("File {0} not exists".format(filename))
        return None
    
    return cv2.imread(filename, flags)

def draw_rectangle(image, rectangle, center_with_size=False, color=[0, 255, 0], thickness=2):
    if center_with_size:
        cx, cy, w, h = rectangle
        left, right = int(cx - w/2), int(cx + w/2)
        top, bottom = int(cy - h/2), int(cy + h/2)
    else:
        left, top, right, bottom = rectangle
        
    return cv2.rectangle(image,(left,top),(right, bottom),color=color,thickness=thickness)

def draw_bounding_box(image, bound_box, center_with_size=True, color=(0,255,0), thickness=2):
    result = draw_rectangle(image, bound_box, center_with_size, color, thickness)
    return result

def draw_annotations(filename, annotations):
    image = load_image(filename, cv2.IMREAD_COLOR)
    if image is None:
        return    
        
    for bound_box in annotations:
        draw_bounding_box(image, bound_box, center_with_size=False)
    return image

def show_image(image):
    cv2.imshow('Image', image)
    while cv2.getWindowProperty('Image', 0) >= 0 :
        val = cv2.waitKey(100)
        if val != 255:
            break
    cv2.destroyWindow('Image') 

