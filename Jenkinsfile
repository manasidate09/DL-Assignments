pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    stages {
        stage('Clone') {
            steps {
                git credentialsId: 'd79b2232-51b6-43ea-8482-94bdb7ac6d1f', url: 'https://github.com/manasidate09/DL-Assignments.git'
            }
        }

        stage('Setup Python') {
            steps {
                sh 'python3 -m venv $VENV_DIR'
                sh './$VENV_DIR/bin/pip install --upgrade pip'
                sh './$VENV_DIR/bin/pip install -r requirements.txt'
            }
        }

        stage('Run Tests') {
            steps {
                sh './$VENV_DIR/bin/python -m unittest discover'
            }
        }
    }
}
