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
        hat_coords = []
        vest_coords = []

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
                    cv2.rectangle(frame, (xmin_v , ymin_v), (xmax_v , ymax_v), (0, 255, 0), 2)
                    vest_coords.append(box[3:])

                # Detect helment
                if int(box[1]) == 4:
                    self.logger.info("helment coordinates: {0}".format(box[1]))
                    xmin_h = int(box[3] * width)
                    ymin_h = int(box[4] * height)
                    xmax_h = int(box[5] * width)
                    ymax_h = int(box[6] * height)
                    cv2.rectangle(frame, (xmin_h , ymin_h), (xmax_h , ymax_h), (0, 255, 0), 2)
                    hat_coords.append(box[3:])
        
        return vest_coords, hat_coords