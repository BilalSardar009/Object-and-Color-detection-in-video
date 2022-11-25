import cv2
import gradio as gr
thres = 0.45 # Threshold to detect object


def Detection(filename):
  cap = cv2.VideoCapture(filename)


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
    try:
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
    except:
        pass
    print(classIds,bbox)
    try:
      if len(classIds) != 0:
          for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
    
              cv2.rectangle(img,box,color=(0,255,0),thickness=2)
              cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
              cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
              cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
              cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
              if FinalItems.count(classNames[classId-1]) == 0:
                FinalItems.append(classNames[classId-1])
            
      
      cv2.imshow("Output",img)
      cv2.waitKey(10)
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
                        title='Object Detection in Video')
interface.launch()