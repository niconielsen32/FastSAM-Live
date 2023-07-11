from fastsam import FastSAM, FastSAMPrompt
import torch
import numpy as np
import cv2
import time


model = FastSAM('FastSAM-x.pt')


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(DEVICE)


cap = cv2.VideoCapture(1)

while cap.isOpened():
    
    suc, frame = cap.read()
    
    start = time.perf_counter()
    

    everything_results = model(
        source=frame,
        device=DEVICE,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
    )
    
    print(everything_results[0].masks.shape)
    print(everything_results[0].boxes.shape)
    print(everything_results[0].boxes[0].xyxy.cpu().numpy())
    
    #for box in everything_results[0].boxes:
        #box = box.xyxy.cpu().numpy()[0]
        #print(box)
        #cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

    #everything_results = list(everything_results)
    
    #print(everything_results[0].masks.shape)
    
    end = time.perf_counter()
    total_time = end - start
    fps = 1 / total_time

    #prompt_process = FastSAMPrompt(frame, everything_results, device=DEVICE)
    
    

    # # everything prompt
    #ann = prompt_process.everything_prompt()
    
    
    # # bbox prompt
    # # bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
    # bboxes default shape [[0,0,0,0]] -> [[x1,y1,x2,y2]]
    # ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])
    # ann = prompt_process.box_prompt(bboxes=[[200, 200, 300, 300], [500, 500, 600, 600]])

    # # text prompt
    #ann = prompt_process.text_prompt(text='cream colored chair')

    # # point prompt
    # # points default [[0,0]] [[x1,y1],[x2,y2]]
    # # point_label default [0] [1,0] 0:background, 1:foreground
    # ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])

    # point prompt
    # points default [[0,0]] [[x1,y1],[x2,y2]]
    # point_label default [0] [1,0] 0:background, 1:foreground
    #ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])
    
    
    

    #img = prompt_process.plot_to_result(frame, annotations=ann)
    
    
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    #cv2.imshow('img', img)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()
