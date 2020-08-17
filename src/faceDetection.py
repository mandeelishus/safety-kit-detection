'''
Face detection class
'''
from model import Model_X
import cv2

class FaceDetection(Model_X):
    '''
    Face Detection Model child class with Model_X as parent class.
    '''
    def __init__(self, model_name, device, extensions):
        super().__init__(model_name, device, extensions)
        
    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        # preprocess the image
        self.logger.info("preprocess input and start inference")
        
        p_image = self.preprocess_input(image)
        # start synchronous inference for specified request
        self.net.start_async(request_id=0, inputs={self.input_name: p_image})
        
        # wait for the result
        if self.net.requests[0].wait(-1) == 0:
            # get the output of the inference
            self.logger.info("Waiting for output of inference")
            outputs=self.net.requests[0].outputs[self.output_name]

            #return self.crop_output(outputs,image)
            return self.preprocess_output(outputs)
    
    '''
    def crop_output(self, outputs, image):
        height = image.shape[0]
        width = image.shape[1]
        coords=[]
        for box in outputs[0][0]:
            conf = box[2]
            if conf > 0.5:    
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                #cv2.imwrite("facesdetected.jpg", image)
                coords=[xmin,ymin,xmax,ymax]
                image = image[ymin:ymax,xmin:xmax]  
            
        return image, coords
    '''
    def preprocess_output(self,outputs):
        flag = False
        coords = []
        for box in outputs[0][0]:
            conf = box[2]
            if conf > 0.5:
                flag = True
                coords.append(box[3:])
        return coords, flag