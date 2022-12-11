# Object Tracker- Sports Analysis
Utilize yolov5 to perform realtime analysis on sports games. Simply specify a simple configuration defining the physics of the video clip, and the ObjectTracker will annotate and track objects.

## Features:
- Tracks physics, including velocities
- Object validation
- Accounts for frame shake/shift
- Can annotate for multiple team classes
- Flexible enough to work across a variety of sports


```
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
      'color':'average',
      'marker':'ellipse',
      'show_velocity':True
    }
  }
})
sports_game.process()
```

<img src="sample.gif" style="width:60%;">
