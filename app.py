import cv2
import gradio as gr
import fast_colorthief
import webcolors
from PIL import Image
thres = 0.45 # Threshold to detect object



def Detection(filename):
  cap = cv2.VideoCapture(filename)
  framecount=0  

  cap.set(3,1280)
  cap.set(4,720)
  cap.set(10,70)

  error="in function 'cv::imshow'"
  classNames= []
  FinalItems=[]
  classFile = 'coco.names'
  with open(classFile,'rt') as f:
    #classNames = f.read().rstrip('n').split('n')
    classNames = f.readlines()


  # remove new line characters
  classNames = [x.strip() for x in classNames]
  print(classNames)
  configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
  weightsPath = 'frozen_inference_graph.pb'


  net = cv2.dnn_DetectionModel(weightsPath,configPath)
  net.setInputSize(320,320)
  net.setInputScale(1.0/ 127.5)
  net.setInputMean((127.5, 127.5, 127.5))
  net.setInputSwapRB(True)

  while True:
    success,img = cap.read()


    
    # #Colour
    try:
      image = Image.fromarray(img)
      image = image.convert('RGBA')
      image = np.array(image).astype(np.uint8)
      palette=fast_colorthief.get_palette(image)
    

      for i in range(len(palette)):
        diff={}
        for color_hex, color_name in webcolors.CSS3_HEX_TO_NAMES.items():
          r, g, b = webcolors.hex_to_rgb(color_hex)
          diff[sum([(r - palette[i][0])**2,
                    (g - palette[i][1])**2,
                    (b - palette[i][2])**2])]= color_name
        if FinalItems.count(diff[min(diff.keys())])==0:
          FinalItems.append(diff[min(diff.keys())])

    except:
      pass
        
    try:
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
    except:
        pass
    print(classIds,bbox)
    try:
      if len(classIds) != 0:
          for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
    
              #cv2.rectangle(img,box,color=(0,255,0),thickness=2)
              #cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
              #cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
              #cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
              #cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
              if FinalItems.count(classNames[classId-1]) == 0:
                FinalItems.append(classNames[classId-1])
            
      
      #cv2.imshow("Output",img)
      cv2.waitKey(10)
      if framecount>cap.get(cv2.CAP_PROP_FRAME_COUNT):
          break
      else:
          framecount+=1
    except  Exception as err:
      print(err)
      t=str(err)
      if t.__contains__(error):
        break

  print(FinalItems)
  return str(FinalItems)

interface = gr.Interface(fn=Detection, 
                        inputs=["video"],
                         outputs="text", 
                        title='Object & Color Detection in Video')
interface.launch(inline=False,debug=True)