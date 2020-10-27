FROM openvino/ubuntu18_dev 

USER root

WORKDIR /app


RUN apt-get update && \
    apt-get install -y --no-install-recommends libpython3.7 libxext6 libsm6 libxrender1 libfontconfig1 libcurl4-openssl-dev python3.7 python3-pip libglib2.0-0

COPY requirements.txt ./
RUN pip3 install setuptools
RUN pip3 install -r requirements.txt

COPY . .

RUN echo "/bin/bash -c 'source /opt/intel/openvino/bin/setupvars.sh'" >> /root/.bashrc

CMD [ "python3", "src/main.py -i 0 -m_g ./models/safety_model/worker_safety_mobilenet -m_p ./models/person_model/FP16/person-detection-retail-0013 -m_f ./models/face_model/face-detection-adas-binary-0001 -m_m ./models/mask_model/face_mask -d CPU -l 0.6" ]
