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


    def predict(self, frame):
        '''
        The safety gear model uses this function to make perdictions on input images/videos
        '''
        # preprocess the image 
        self.logger.info("preprocess image and start inference")
        
        p_image = self.preprocess_input(frame)

        # start inference for specified request
        # result = self.exec_net(request_id, p_image, infer_handle)
        # start asynchronous inference for specified request
        if p_image.shape[0] is not 0 and p_image.shape[1] is not 0 and p_image.shape[2] is not 0:
            self.net.start_async(request_id=0, inputs={self.input_name: p_image})
        
        # wait for the result
        if self.net.requests[0].wait(-1) == 0:
            # get the output of the inference
            self.logger.info("Waiting for output of inference")
            outputs=self.net.requests[0].outputs[self.output_name]
            
            
            self.logger.info("cropped Person: {0}".format(outputs))


            # select coords based on confidence threshold
            return self.denorm_output(outputs)
        

    def denorm_output(self, result):
        '''
        Before feeding the output of this model to the next model,
        you might have to extract the output. This function is where you can do that.
        '''
        vest_flag = False
        helment_flag = False
        # filter output based on confidence threshold
        hat_coords = []
        vest_coords = []

        for box in result[0][0]:

            conf = box[2]
            if (conf > 0.6):

                # Detect safety vest
                if (int(box[1])) == 2:
                    vest_flag = True
                    self.logger.info("vest coordinates: {0}".format(box[1]))
                    vest_coords.append(box[3:])

                # Detect helment
                if int(box[1]) == 4:
                    helment_flag = True
                    self.logger.info("helment coordinates: {0}".format(box[1]))
                    hat_coords.append(box[3:])
        
        return vest_flag, helment_flag, vest_coords, hat_coords