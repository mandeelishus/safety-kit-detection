'''
Base Class for safety apparel detection models
'''
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import cv2
import sys
import logging 

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

class Model_X:
    '''
    Parenr Class for the computer vision models.
    '''
    def __init__(self, model_name, device='CPU', extensions=CPU_EXTENSION):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.cpu_extension=extensions
        self.logger = logging.getLogger(__name__)

        try:
            self.model=IENetwork(self.model_structure,self.model_weights)
        except Exception as e:
            self.logger.exception("Could not initialize the Network. Have you entered the correct model path?")
        

    def load_model(self, name=None):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        # initialize the IECore interface
        self.core = IECore()

        ### TODO: Check for supported layers ###
        supported_layers = self.core.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers)!=0:
            ### TODO: Add any necessary extensions ###
            if self.cpu_extension and "CPU" in self.device:
                self.core.add_extension(self.cpu_extension, self.device)
            else:
                self.logger.debug("Add CPU extension and device type or run layer with original framework")
                exit(1)

        # load the model
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)

        if name is not None:
            self.input_name=[i for i in self.model.inputs.keys()]
            self.input_shape=self.model.inputs[self.input_name[1]].shape
        else:
            self.input_name=next(iter(self.model.inputs))
            self.input_shape=self.model.inputs[self.input_name].shape

        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

        return 

    def exec_net(self, p_image, request_id, infer_handle):
        '''
        start asynchronous inference for specificed request.
        :param request_id :index of infer request value. Limited to device capabilities.
        :return: Instance of Executable Network class
        '''
        # start asynchronous inference for specified request
        if infer_handle == "async":
            self.net.start_async(request_id=request_id,inputs={self.input_name: p_image})

        # start synchronous inference for specified request
        elif infer_handle == "sync":
            self.net.infer(inputs=p_image)

        # wait for the result
        if self.wait(request_id) == 0:
            # get the output of the inference
            self.logger.info('waiting for output of inference')
            return self.get_output(request_id)

    def wait(self, request_id):
        """
        Waits for the result to become available.
        :param request_id: Index of Infer request value. Limited to device capabilities.
        :return: Timeout value
        """
        return self.net.requests[request_id].wait(-1)

    def get_output(self, request_id):
        """
        Gives a list of results for the output layer of the network.
        :param request_id: Index of Infer request value. Limited to device capabilities.
        :param output: Name of the output layer
        :return: Results for the specified request
        """
        return self.net.requests[request_id].outputs[self.output_name]
        

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        dsize = (self.input_shape[3], self.input_shape[2])
        curr_image=image
        try:
            curr_image = cv2.resize(image,(dsize))
            curr_image = curr_image.transpose((2,0,1))
            curr_image = curr_image.reshape(1,*curr_image.shape)
        except Exception as e:
            self.logger.info(str(e))
        return curr_image