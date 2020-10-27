pipeline {
    agent none 
    stages {
        stage('Build') { 
            agent { dockerfile true}
            steps {
                sh 'python3 -m py_compile src/faceDetection.py src/faceMaskDetection.py src/main.py src/model.py src/personDetection.py src/safetyGear.py src/input_feeder.py' 
                stash(name: 'compiled-results', includes: 'src/*.py*') 
            }
        }
    }
}