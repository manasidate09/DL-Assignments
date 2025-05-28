pipeline {
    agent any

    stages {
        stage('Clone Repository') {
            steps {
                git 'https://github.com/manasidate09/DL-Assignment.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Run Tests') {
            steps {
                sh 'pytest tests/' // update this path as needed
            }
        }

        stage('Train Model') {
            steps {
                sh 'python app.py' // replace with your actual script
            }
        }

        stage('Archive Results') {
            steps {
                archiveArtifacts artifacts: '**/*.pkl', allowEmptyArchive: true
            }
        }
    }

    post {
        always {
            echo 'Pipeline completed'
        }
        failure {
            echo 'Pipeline failed'
        }
    }
}
