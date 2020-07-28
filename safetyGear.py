'''
safety gear detection class.
'''
from model import Model_X
import cv2


class GearDetect(Model_X):
    '''
    Safety gear detection model class
    '''
    def __init__(self, model_name, device, extensions):
        super().__init__(model_name, device, extensions)


    def predict(self, frame, request_id, infer_handle):
        '''
        The safety gear model uses this function to make perdictions on input images/videos
        '''
        # preprocess the image 
        self.logger.info("preprocess image and start inference")
        
        p_image = self.preprocess_input(frame)

        # start inference for specified request
        result = self.exec_net(request_id, p_image, infer_handle)

        # select coords based on confidence threshold
        return self.denorm_output(result)
        

    def denorm_output(self, result, frame):
        '''
        Before feeding the output of this model to the next model,
        you might have to extract the output. This function is where you can do that.
        '''
        height = frame.shape[0]
        width = frame.shape[1]
        
        # filter output based on confidence threshold
        vest_detect_coord = []
        helment_detect_coord = []

        for box in result[0][0]:

            conf = box[2]
            if (conf > 0.6):

                # Detect safety vest
                if (int(box[1])) == 2:
                    self.logger.info("vest coordinates: {0}".format(box[1]))
                    xmin_v = int(box[3] * width)
                    ymin_v = int(box[4] * height)
                    xmax_v = int(box[5] * width)
                    ymax_v = int(box[6] * height)

                    vest_detect_coord.append([xmin_v,ymin_v,xmax_v,ymax_v])

                # Detect helment
                if int(box[1]) == 4:
                    self.logger.info("helment coordinates: {0}".format(box[1]))
                    xmin_h = int(box[3] * width)
                    ymin_h = int(box[4] * height)
                    xmax_h = int(box[5] * width)
                    ymax_h = int(box[6] * height)

                    helment_detect_coord.append([xmin_h,ymin_h,xmax_h,ymax_h])  

            for vest, helment in  (vest_detect_coord, helment_detect_coord):
                cv2.rectangle(frame, (vest[0] , vest[1]), (vest[2] , vest[3]), (0, 255, 0), 2)
                cv2.rectangle(frame, (helment[0] , helment[1]), (helment[2] , helment[3]), (0, 255, 0), 2)
        return (vest_detect_coord, helment_detect_coord, frame)          