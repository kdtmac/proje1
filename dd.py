import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import argparse
import sys
import json
import numpy as np
import time

def main(args):
    mode = args.mode
    if(mode == "camera"):
        zhaoyizhao()
    elif mode == "input":
        create_manual_data();
    else:
        raise ValueError("Unimplemented mode")

# #现在需要为图片编号的(全局)变量
# int bim = 0     #这个标志新人脸的编号
# int frnum = 0   #这个标志当前正在处理帧的编号

def camera_recog():
    print("[INFO] camera sensor warming up...")
    vs = cv2.VideoCapture(0); #get input from webcam
    while True:
        _,frame = vs.read();
        #u can certainly add a roi here but for the sake of a demo i'll just leave it as simple as this
        rects, landmarks = face_detect.detect_face(frame,80);#min face size is set to 80x80
        aligns = []
        positions = []
        for (i, rect) in enumerate(rects):
            aligned_face, face_pos = aligner.align(160,frame,landmarks[i])
            cv2.imshow('face',aligned_face)
            cv2.imwrite('1'+str(i)+'.jpeg',aligned_face)
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                aligns.append(aligned_face)
                positions.append(face_pos)
            else: 
                print("Align face failed") #log
        features_arr = extract_feature.get_features(aligns)
        recog_data = findPeople(features_arr,positions);
        for (i,rect) in enumerate(rects):
            cv2.rectangle(frame,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(255,0,0)) #draw bounding box for the face
            cv2.putText(frame,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)

        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break    

def zhaoyizhao():
    #现在需要为图片编号的(全局)变量
    #这个标志新人脸的编号
    bim = 0
    #这个标志当前正在处理帧的编号
    frnum = 0

    # f = open('/home/agnes/FaceRec-master/facerec_128D.txt','r');
    # data_set = json.loads(f.read());
    print("[INFO] camera sensor warming up...")
    vs = cv2.VideoCapture(0); #get input from webcam
    while True:
        frnum = frnum +1
        bim = 1
        _,frame = vs.read();
        #u can certainly add a roi here but for the sake of a demo i'll just leave it as simple as this
        rects, landmarks = face_detect.detect_face(frame,80);#min face size is set to 80x80
        aligns = []
        positions = []
        for (i, rect) in enumerate(rects):
            aligned_face, face_pos = aligner.align(160,frame,landmarks[i])
            cv2.imshow('face',aligned_face)
            cv2.imwrite('1'+str(i)+'.jpeg',aligned_face)
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                aligns.append(aligned_face)
                positions.append(face_pos)
            else: 
                print("Align face failed") #log

    #以上行是把人脸抠出来，存进了aligns中
        features_arr = extract_feature.get_features(aligns)
        recog_data = findPeople(features_arr,positions);     
    #把识别数据存进recog_data里，其中recog_data[i][0]是姓名或unknown，recog_data[i][1]是识别相似度百分比。
    #这时候要判断是不是unknown，如果是，存储进去，如果不是，删除该人脸数据。
        for (i, rect) in enumerate(rects):
            if (recog_data[i][0]=='Unknown'):
                #
                cv2.imwrite(str(frnum)+"0"+'.jpeg',frame)
                #存储大图
                #这里要存储了，需要调用aligns中对应序号的那个人脸，输入进函数（其实所有要用到的参数已经获得，只需要顺序i就可以了
                #首先，读取数据库，并且新建模板准备存储
                f = open('/home/agnes/FaceRec-master/facerec_128D.txt','r');
                data_set = json.loads(f.read());
                person_imgs = {"Left" : [], "Right": [], "Center": []};
                person_features = {"Left" : [], "Right": [], "Center": []};
                #从刚才的比对中找到那个新脸，开始存储
                aligned_face = aligns[i]
                if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                            person_imgs[face_pos].append(aligned_face)
                            cv2.imshow("Captured face", aligned_face)
                            #
                            cv2.imwrite(str(frnum)+str(bim)+'.jpeg',aligned_face)
                            #存储新人脸图片
                for face_pos in person_imgs: #there r some exceptions here, but I'll just leave it as this to keep it simple
                    person_features[face_pos] = [np.mean(extract_feature.get_features(person_imgs[face_pos]),axis=0).tolist()]
                data_set[str(frnum)+str(bim)] = person_features;
                f = open('/home/agnes/FaceRec-master/facerec_128D.txt', 'w');
                f.write(json.dumps(data_set))                
            else: 
                f = open('/home/agnes/FaceRec-master/facerec_128D.txt','r');
                data_set = json.loads(f.read());
                del data_set[recog_data[i][0]]
                f = open('/home/agnes/FaceRec-master/facerec_128D.txt', 'w');
                f.write(json.dumps(data_set))  

        
            cv2.rectangle(frame,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(255,0,0)) #draw bounding box for the face
            cv2.putText(frame,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
        time.sleep(1)
        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
             break 


def findPeople(features_arr, positions, thres = 0.5, percent_thres = 70): # thres原来是0.6
    '''
    :param features_arr: a list of 128d Features of all faces on screen
    :param positions: a list of face position types of all faces on screen
    :param thres: distance threshold
    :return: person name and percentage
    '''
    f = open('/home/agnes/FaceRec-master/facerec_128D.txt','r')
    data_set = json.loads(f.read());
    returnRes = [];
    for (i,features_128D) in enumerate(features_arr):
        result = "Unknown";
        smallest = sys.maxsize
        for person in data_set.keys():
            person_data = data_set[person][positions[i]];
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data-features_128D)))
                if(distance < smallest):
                    smallest = distance;
                    result = person;
        percentage =  min(100, 100 * thres / smallest)
        if percentage <= percent_thres :
            result = "Unknown"
        returnRes.append((result,percentage))
    return returnRes
          


'''
#下面总结存储一个新数据需要什么

def createnew(int i):
    #首先，读取数据库，并且新建模板准备存储
    f = open('/home/agnes/FaceRec-master/facerec_128D.txt','r');
    data_set = json.loads(f.read());
    person_imgs = {"Left" : [], "Right": [], "Center": []};
    person_features = {"Left" : [], "Right": [], "Center": []};
    #从刚才的比对中找到那个新脸，开始存储
    aligned_face = aligns[i]
    if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                person_imgs[pos].append(aligned_face)
                cv2.imshow("Captured face", aligned_face)
    for face_pos in person_imgs: #there r some exceptions here, but I'll just leave it as this to keep it simple
        person_features[face_pos] = [np.mean(extract_feature.get_features(person_imgs[face_pos]),axis=0).tolist()]
    data_set[new_name] = person_features;
    f = open('/home/agnes/FaceRec-master/facerec_128D.txt', 'w');
    f.write(json.dumps(data_set))                

'''


def create_manual_data():
    vs = cv2.VideoCapture(0); #get input from webcam
    
    print("Please input new user ID:")
    new_name = input(); #ez python input()
    f = open('/home/agnes/FaceRec-master/facerec_128D.txt','r');
    data_set = json.loads(f.read());
    person_imgs = {"Left" : [], "Right": [], "Center": []};
    person_features = {"Left" : [], "Right": [], "Center": []};
    print("Please start turning slowly. Press 'q' to save and add this new user to the dataset");
    while True:
        _, frame = vs.read();
        rects, landmarks = face_detect.detect_face(frame, 80);  # min face size is set to 80x80
        for (i, rect) in enumerate(rects):
            aligned_frame, pos = aligner.align(160,frame,landmarks[i]);
            if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
                person_imgs[pos].append(aligned_frame)
                cv2.imshow("Captured face", aligned_frame)
                cv2.imwrite('123.jpeg',aligned_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    for pos in person_imgs: #there r some exceptions here, but I'll just leave it as this to keep it simple
        person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]),axis=0).tolist()]
    data_set[new_name] = person_features;
    f = open('/home/agnes/FaceRec-master/facerec_128D.txt', 'w');
    f.write(json.dumps(data_set))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Run camera recognition", default="camera")
    args = parser.parse_args(sys.argv[1:]);
    FRGraph = FaceRecGraph();
    aligner = AlignCustom();
    extract_feature = FaceFeature(FRGraph)
    face_detect = MTCNNDetect(FRGraph, scale_factor=2); #scale_factor, rescales image for faster detection
    main(args);