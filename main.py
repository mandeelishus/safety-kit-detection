import numpy as np
import cv2
import argparse
import logging
from input_feeder import InputFeeder
from faceDetection import FaceDetection
from faceMaskDetection import MaskDetection
from personDetection import PersonDetect
from safetyGear import GearDetect
import time
import os

CPU_EXTENSION="/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
performance_directory_path="../"

def get_args():
    '''
    This function gets the arguments from the command line.
    '''

    parser = argparse.ArgumentParser()
    
    # --Add required and optional groups
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # --create the arguments
    optional.add_argument("-m_f", help="path to face detection model", default=None)
    optional.add_argument("-m_m", help="path to mask detection model", default=None)
    optional.add_argument("-m_p", help="path to person detection model", default=None)
    optional.add_argument("-m_g", help="path to safety gear detection model", default=None)

    required.add_argument("-m_f",help="path to face detection model", required=True)
    required.add_argument("-m_m",help="path to mask detection model", required=True)
    required.add_argument("-m_p",help="path to person detection model", required=True)
    required.add_argument("-m_g",help="path to safety gear detection model", required=True)
    required.add_argument("-i", help="path to video/image file or 'cam' for webcam", required=True)

    optional.add_argument("-l", help="MKLDNN (CPU)-targeted custom layers.", default=CPU_EXTENSION, required=False)
    optional.add_argument("-d", help="Specify the target device type", default='CPU')
    optional.add_argument("-p", help="path to store performance stats", required=False)
    optional.add_argument("-vf", help="specify flags from m_f, m_m, m_p, m_g e.g. -vf m_f m_m m_p m_g (seperate each flag by space) for visualization of the output of intermediate models", nargs='+', default=[], required=False)

    args = parser.parse_args()
    return args

def writePerformanceStats(args,model_stats_txt,model_inference_time,model_fps,model_load_time):
    '''
    This function writes perfomance status of each model used to a txt file.

    :param arg:
    :param model_stats_txt: 
    :param  : 
    :param  : 
    '''
    with open(os.path.join(performance_directory_path+args.p, model_stats_txt), 'w') as f:
                f.write(str(model_inference_time)+'\n')
                f.write(str(model_fps)+'\n')
                f.write(str(model_load_time)+'\n')


def pipelines(args):
    '''
    This function writes perfomance status of each model used to a txt file.
    '''
    # enable logging for the function
    logger = logging.getLogger(__name__)
    
    # grab the parsed parameters
    faceDetectionModel=args.m_f
    maskDetectionModel=args.m_m
    personDetectionModel=args.m_p
    gearDetectionModel=args.m_g
    device=args.d
    customLayers=args.l
    inputFile=args.i
    visualization_flag = args.vf

        # initialize feed
    single_image_format = ['jpg','tif','png','jpeg', 'bmp']
    if inputFile.split(".")[-1].lower() in single_image_format:
        feed=InputFeeder('image',inputFile)
    elif args.i == 'cam':
        feed=InputFeeder('cam')
    else:
        feed = InputFeeder('video',inputFile)

    # load feed data
    feed.load_data()

        # initialize and load the models
    start_face_model_load_time = time.time()
    faceDetectionPipeline=FaceDetection(faceDetectionModel, device, customLayers)
    faceDetectionPipeline.load_model()
    face_model_load_time = time.time() - start_face_model_load_time

    start_mask_model_load_time = time.time()
    maskDetectionPipeline=MaskDetection(maskDetectionModel, device, customLayers)
    maskDetectionPipeline.load_model()
    mask_model_load_time = time.time() - start_mask_model_load_time
    
    start_person_model_load_time = time.time()
    personDetectionPipeline=PersonDetect(personDetectionModel, device, customLayers)
    personDetectionPipeline.load_model()
    person_model_load_time = time.time() - start_person_model_load_time
    
    start_gear_model_load_time = time.time()
    gearDetectionPipeline=GearDetect(gearDetectionModel, device, customLayers)
    gearDetectionPipeline.load_model()
    gear_model_load_time = time.time() - start_gear_model_load_time

     # set framecount, request_id, infer_handle variables
    frameCount = 0
    request_id = 0
    infer_async = "async"
    infer_sync = "sync"
    # collate frames from the feeder and feed into the detection pipelines
    for _, frame in feed.next_batch():

        if not _:
            break
        frameCount+=1
        #if frameCount%5==0:
            #cv2.imshow('video', cv2.resize(frame,(500,500)))

        key = cv2.waitKey(60)

        start_person_inference_time = time.time()
        croppedperson = personDetectionPipeline.predict(frame)
        person_inference_time = time.time() - start_person_inference_time

        if 'm_p' in visualization_flag:
            cv2.imshow('cropped person', croppedperson)

        if type(croppedperson)==int:
            logger.info("no person detected")
            if key==27:
                break
            continue

        start_gear_inference_time = time.time()
        vest_detect, helment_detect, person = gearDetectionPipeline.predict(croppedperson.copy(), request_id, infer_sync)
        gear_inference_time = time.time() - start_gear_inference_time

        if 'm_g' in visualization_flag:
            cv2.imshow('gear output', person)
            # cv2.putText()

        start_face_inference_time = time.time()
        face coord, f_image=faceDetectionPipeline.predict(croppedperson.copy())   
        face_inference_time = time.time() - start_face_inference_time

        if 'm_f' in visualization_flag:
            # cv2.imshow('face', f_image)

        start_mask_inference_time = time.time()
        mask_detect = maskDetectionPipeline.predict(f_image)
        mask_inference_time = time.time() - start_mask_inference_time

        if 'm_m' in visualization_flag:
            cv2.imshow('mask', person)
            # cv2.putText()
        
        if key==27:
            break
            
    logger.info("The End")
    cv2.destroyAllWindows()
    feed.close()

def main():
    args=get_args()
    pipelines(args) 

if __name__ == '__main__':
    main()
