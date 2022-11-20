from imageai.Detection import ObjectDetection
import os
from PIL import Image, ImageDraw
import cv2
import math 

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()
# custom_objects = detector.CustomObjects(car=True, motorcycle=True)
SAMPLE_RATE = 2



class Object:
  def __init__(self, cx, cy):
    self.xs = [cx]
    self.ys = [cy]
    self.vxs = [0]
    self.vys = [0]
    self.oxs = [0]
    self.oys = [0]
    

  def update_nearest(self, detections):
    best={
      'distance':1000000,
      'x':0,
      'y':0
    }
    for detection in detections:
      cx=(detection.get('box_points')[0]+detection.get('box_points')[2])/2
      cy=(detection.get('box_points')[1]+detection.get('box_points')[3])/2
    
      distance = math.dist((self.xs[-1],self.ys[-1]),(cx,cy))
      if distance<best.get('distance'):
        # Closest point
        best = {
          'distance':distance,
          'x':cx,
          'y':cy
        }
    

    # Update locations
    self.xs.append(best.get('x'))
    self.ys.append(best.get('y'))
    # Update velocities 
    self.vxs.append(self.xs[-1]-self.xs[-2])
    self.vys.append(self.xs[-1]-self.xs[-2]) 

class ObjectTracker:
  def __init__(self, video):
    self.video = video
    # Detection labels we want to track
    self.labels = {
      'ball':{
        'labels':[
          'sports_ball'
        ],
        'required_confidence':0.5,
        'allowed_distance':0.5,
        'predict_path':'line'
      },
      'person':{
        'labels':[
          'person'
        ],
        'required_confidence':0.5,
        'allowed_distance':0.5,
        'predict_path':'line',
      }
    }
    self.objects = []

  def update_objects(self,detections):
    if len(self.objects)==0:
      for detection in detections:
        cx=(detection.get('box_points')[0]+detection.get('box_points')[2])/2
        cy=(detection.get('box_points')[1]+detection.get('box_points')[3])/2
        self.objects.append(Object(cx,cy))
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
        obj.oxs.append(avg_offset_x)
        obj.oys.append(avg_offset_y)
        


  def process(self):
    vidcap = cv2.VideoCapture(self.video)
    count = 0
    frame_count = 0
    success = True

    while success:
      success,image = vidcap.read()
      if count % SAMPLE_RATE == 0:
        print(count)
      
        image_path = f"./output/frame{frame_count}"
        cv2.imwrite(image_path+".jpg", image)     # save frame as JPEG file      
        

        detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , image_path+".jpg"), output_image_path=os.path.join(execution_path , image_path+"_detected.jpg"), custom_objects = detector.CustomObjects(person=True, sports_ball=True), minimum_percentage_probability=15)

        self.update_objects(detections)
        # Annotate image
        image = Image.open(image_path+".jpg")
        draw = ImageDraw.Draw(image)
        for object in self.objects:
          for i in range(0,len(object.xs)):
            x=object.xs[i]+object.oxs[i]
            y=object.ys[i]+object.oys[i]
            draw.rectangle((x, y,x+2,y+2), width=5, fill='blue')
          draw.rectangle((x, y,x+object.vxs[-1],y+object.vys[-1]), width=5, fill='blue')
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