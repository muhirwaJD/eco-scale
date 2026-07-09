// Simple CI/CD pipeline: build the Docker image and push it to a registry.
//
// Prerequisites (one-time Jenkins setup):
//   1. The agent has Docker installed and the `jenkins` user can run it.
//   2. A "Username with password" credential (Docker Hub username + access token)
//      stored in Jenkins with the ID  dockerhub  (see DOCKER_CRED below).
//   3. Change IMAGE to your own Docker Hub repo (user/name).

pipeline {
  agent any

  environment {
    IMAGE       = 'yourdockerhubuser/eco-scale'   // <-- change to your Docker Hub repo
    DOCKER_CRED = credentials('dockerhub')        // binds DOCKER_CRED_USR / DOCKER_CRED_PSW
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
