'''
Person detection class
'''
from model import Model_X
import cv2

class MaskDetection(Model_X):
    '''
    Person/people detection Model child class with Model_X as parent class.
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
        # start asynchronous inference for specified request
        if p_image.shape[0] is not 0 and p_image.shape[1] is not 0 and p_image.shape[2] is not 0:
            self.net.start_async(request_id=0, inputs={self.input_name: p_image})
        
        # wait for the result
        if self.net.requests[0].wait(-1) == 0:
            # get the output of the inference
            self.logger.info("Waiting for output of inference")
            outputs=self.net.requests[0].outputs[self.output_name]
            
            
            self.logger.info("cropped Person: {0}".format(outputs))
            return self.preprocess_output(outputs)

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # filter output based on confidence threshold
        result = outputs[0][0]
        
        return result