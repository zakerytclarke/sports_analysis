from imageai.Detection import ObjectDetection
import os
from PIL import Image, ImageDraw
import cv2
import math 
import torch
import numpy as np
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.01

execution_path = os.getcwd()

os.system("rm -rf output && mkdir output")
os.system("rm -rf runs")

SAMPLE_RATE = 1

MAX_DISTANCE = 30


def getDetections(raw, configs):
  out = []
  names=raw.names

  df = raw.pandas().xyxy[0]

  df['objects'] = df.apply(lambda row: {
        'name':names[row['class']],
        'box_points':[row.xmin,row.ymin,row.xmax,row.ymax],
        'confidence':row.confidence
      },axis=1)
  for obj in df['objects']:
    if obj.get('name') in configs and obj.get('confidence') >= configs[obj.get('name')].get('confidence'):
      out.append(obj)

  return out

class Object:
  def __init__(self, label, cx, cy, box_points, config):
    # Position
    self.xs = [cx]
    self.ys = [cy]

    self.box_points = [box_points]
    # Velocity
    self.vxs = [0]
    self.vys = [0]
    # Offset
    self.oxs = [0]
    self.oys = [0]
    # Cumulative offset
    self.coxs = [0]
    self.coys = [0]


    self.label = label
    self.disabled = False # Whether object is in frame 
    
    self.config = config

  def update_nearest(self, detections):
    best={
      'distance':1000000,
      'x':0,
      'y':0
    }
    # Search through objects with same label
    for detection in filter(lambda x:x.get('name')==self.label, detections):
      cx=(detection.get('box_points')[0]+detection.get('box_points')[2])/2
      cy=(detection.get('box_points')[1]+detection.get('box_points')[3])/2
    
      distance = math.dist((self.xs[-1],self.ys[-1]),(cx,cy))
      if distance<best.get('distance'):
        # Closest point
        best = {
          'distance':distance,
          'x':cx,
          'y':cy,
          'box_points':detection.get('box_points'),
          'obj':detection
        }
    

    # Determine if same object
    if best.get('distance') <= self.config.get('max_distance'):
      # Update locations
      self.xs.append(best.get('x'))
      self.ys.append(best.get('y'))
      self.box_points.append(best.get('box_points'))
      # Update velocities 
      self.vxs.append(self.xs[-1]-self.xs[-2])
      self.vys.append(self.xs[-1]-self.xs[-2]) 
      # Flag as belonging to another object
      best.get('obj')['processed'] = True
    else:
      self.disabled = True

    

class ObjectTracker:
  def __init__(self, config):
    self.video = config.get('video')
    assert self.video, "Please provide a video url"
    self.object_settings = config.get('objects')
    # Detection objects we want to track
    self.objects = []

  def update_objects(self,detections):
    if len(self.objects)==0:
      for detection in detections:
        cx=(detection.get('box_points')[0]+detection.get('box_points')[2])/2
        cy=(detection.get('box_points')[1]+detection.get('box_points')[3])/2
        if detection.get('name') in self.object_settings:
          self.objects.append(Object(detection.get('name'),cx,cy,detection.get('box_points'),self.object_settings[detection.get('name')]))
    else:
      for obj in self.objects:
        # Update object locations
        obj.update_nearest(detections) 
      # Update average offset
      tx = list(map(lambda x:x.vxs[-1], self.objects))
      ty = list(map(lambda x:x.vys[-1], self.objects))
      avg_offset_x = sum(tx)/len(tx)
      avg_offset_y = sum(ty)/len(ty)
      for obj in self.objects:
        obj.xs[-1] += avg_offset_x
        obj.ys[-1] += avg_offset_y
        obj.oxs.append(avg_offset_x)
        obj.oys.append(avg_offset_y)
        obj.coxs.append(obj.coxs[-1]+obj.oxs[-1])
        obj.coys.append(obj.coys[-1]+obj.oys[-1])
      # Find unlabeled objects
      new_objs = list(filter(lambda x:not x.get('processed',False),detections))
      for n_obj in new_objs:
        cx=(n_obj.get('box_points')[0]+n_obj.get('box_points')[2])/2
        cy=(n_obj.get('box_points')[1]+n_obj.get('box_points')[3])/2
        if n_obj.get('name') in self.object_settings:
          self.objects.append(Object(n_obj.get('name'),cx,cy,n_obj.get('box_points'),self.object_settings[n_obj.get('name')]))
        


  def process(self):
    vidcap = cv2.VideoCapture(self.video)
    count = 0
    frame_count = 0
    success = True

    while success:
      success,image = vidcap.read()
      
      if success and count % SAMPLE_RATE == 0:
        print(count)
      
        image_path = f"./output/frame{str(frame_count).zfill(5)}"
        cv2.imwrite(image_path+".jpg", image)     # save frame as JPEG file      
        
        results = model(image)
        
        # results.save(image_path+"_detected.jpg")
        
        detections = getDetections(results, self.object_settings)
        
        self.update_objects(detections)
        # Annotate image
        image = Image.open(image_path+".jpg")
        draw = ImageDraw.Draw(image)
        for object in self.objects:
          ox = 0
          oy = 0
          

          if not object.disabled:
            obj_settings = self.object_settings[object.label]
            if obj_settings:
              if obj_settings.get('color')=='average':
                b,g,r = list(np.average(np.average(image.crop(object.box_points[-1]), axis=0), axis=0))
                color = (int(r),int(g),int(b))
              else:
                color = obj_settings.get('color')
                
              marker = obj_settings.get('marker')
              # Draw marker
              if marker=='ellipse':
                draw.ellipse((object.xs[-1]-20, object.ys[-1]-10+20, object.xs[-1]+20, object.ys[-1]+10+20), width=5, outline=color)
              elif marker=='circle':
                draw.ellipse((object.xs[-1]-10, object.ys[-1]-10, object.xs[-1]+10, object.ys[-1]+10), width=5, outline=color)
              elif marker=='box':
                draw.rounded_rectangle(object.box_points[-1], width=3, radius=5, outline=color)
              
              # Draw Velocity
              if 'show_velocity' in obj_settings:
                draw.line((object.xs[-1], object.ys[-1], object.xs[-1] + object.vxs[-1] - object.oxs[-1], object.ys[-1] + object.vys[-1] - object.oys[-1]), width=2, fill=color)

        image.save(image_path+".jpg")
        frame_count += 1
      count += 1
    print("Finished Processing")
    os.system("ffmpeg -r 40 -f image2 -i ./output/frame%05d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p processed.mp4 -y")
    





sports_game = ObjectTracker({
  'video':"./soccer.mp4",
  'objects':{
    'sports ball':{
      'confidence':0.1,
      'max_distance':50,
      'color':'white',
      'marker':'circle',
      'show_velocity':True
    },
    'person':{
      'confidence':0.3,
      'max_distance':30,
      # 'color':'#2bd68c',
      'color':'average',
      'marker':'ellipse',
      'show_velocity':True
    }
  }
})
sports_game.process()
