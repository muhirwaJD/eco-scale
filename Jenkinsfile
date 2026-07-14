pipeline {
  agent any

  environment {
    IMAGE       = 'muhirwajd/eco-scale'   
    DOCKER_CRED = credentials('dockerhub')       
  }

  stages {
    stage('Checkout') {
      steps { checkout scm }
    }

    stage('Build image') {
      steps {
        sh 'docker build -t $IMAGE:$BUILD_NUMBER -t $IMAGE:latest .'
      }
    }

    stage('Push image') {
      steps {
        sh '''
          echo "$DOCKER_CRED_PSW" | docker login -u "$DOCKER_CRED_USR" --password-stdin
          docker push $IMAGE:$BUILD_NUMBER
          docker push $IMAGE:latest
          docker logout
        '''
      }
    }
  }

  post {
    always {
      sh 'docker image prune -f || true'   // free disk on the agent
    }
    success {
      echo "Pushed $IMAGE:latest (build $BUILD_NUMBER)"
    }
  }
}
