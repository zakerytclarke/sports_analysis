from imageai.Detection import ObjectDetection
import os
from PIL import Image, ImageDraw
import cv2
import math 
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


execution_path = os.getcwd()

os.system("rm -rf output && mkdir output")
os.system("rm -rf runs")

# detector = ObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))
# detector.loadModel()
# custom_objects = detector.CustomObjects(car=True, motorcycle=True)
SAMPLE_RATE = 2

MAX_DISTANCE = 30

def getDetections(raw,w,h):
  out = []
  names = raw.names
  
  for tensor in raw.xyxyn[0]:
    box=tensor.tolist()[0:4]
    out.append({
      'name':names[tensor.tolist()[5]],
      'box_points':[w*box[0],h*box[1],w*box[2],h*box[3]]
    })

  return out

class Object:
  def __init__(self, label, cx, cy):
    self.xs = [cx]
    self.ys = [cy]
    self.vxs = [0]
    self.vys = [0]
    self.oxs = [0]
    self.oys = [0]

    self.label = label
    

  def update_nearest(self, detections):
    best={
      'distance':1000000,
      'x':0,
      'y':0
    }
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
          'obj':detection
        }
    

    # Determine if same object
    if best.get('distance') <= MAX_DISTANCE:
      # Update locations
      self.xs.append(best.get('x'))
      self.ys.append(best.get('y'))
      # Update velocities 
      self.vxs.append(self.xs[-1]-self.xs[-2])
      self.vys.append(self.xs[-1]-self.xs[-2]) 
      # Flag as belonging to another object
      best.get('obj')['processed'] = True


    

class ObjectTracker:
  def __init__(self, video):
    self.video = video
    # Detection objects we want to track
    self.objects = []

  def update_objects(self,detections):
    if len(self.objects)==0:
      for detection in detections:
        cx=(detection.get('box_points')[0]+detection.get('box_points')[2])/2
        cy=(detection.get('box_points')[1]+detection.get('box_points')[3])/2
        self.objects.append(Object(detections.get('name'),cx,cy))
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
      # Find unlabeled objects
      new_objs = list(filter(lambda x:not x.get('processed',False),detections))
      for n_obj in new_objs:
        cx=(n_obj.get('box_points')[0]+n_obj.get('box_points')[2])/2
        cy=(n_obj.get('box_points')[1]+n_obj.get('box_points')[3])/2
        self.objects.append(Object(n_obj.get('name'),cx,cy))
        


  def process(self):
    vidcap = cv2.VideoCapture(self.video)
    count = 0
    frame_count = 0
    success = True

    while success:
      success,image = vidcap.read()
      h, w, c = image.shape
      if count % SAMPLE_RATE == 0:
        print(count)
      
        image_path = f"./output/frame{frame_count}"
        cv2.imwrite(image_path+".jpg", image)     # save frame as JPEG file      
        

        # detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , image_path+".jpg"), output_image_path=os.path.join(execution_path , image_path+"_detected.jpg"), custom_objects = detector.CustomObjects(person=True, sports_ball=True), minimum_percentage_probability=15)
        
        
        results = model(image)
        results.save(image_path+"_detected.jpg")
        
        detections = getDetections(results,w,h)
        
        self.update_objects(detections)
        # Annotate image
        image = Image.open(image_path+".jpg")
        draw = ImageDraw.Draw(image)
        for object in self.objects:
          ox = 0
          oy = 0
          import ipdb
          ipdb.set_trace()
          draw.ellipse((object.xs[-1]-20, object.ys[-1]-10+20, object.xs[-1]+20, object.ys[-1]+10+20), width=5, outline='#2bd68c')
        image.save(image_path+"_annotated.jpg")
        frame_count += 1
      count += 1





sports_game = ObjectTracker("./soccer.mp4")
sports_game.process()





# def line(image_path, output_path):
#     image = Image.open(image_path)
#     draw = ImageDraw.Draw(image)
#     colors = ["red", "green", "blue", "yellow",
#               "purple", "orange"]
#     for i in range(0, 100, 20):
#         draw.line((i, 0) + image.size, width=5, 
#                   fill=random.choice(colors))
#     image.show()