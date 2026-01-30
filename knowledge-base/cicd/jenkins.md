# Jenkins Pipeline Troubleshooting

## Common Pipeline Errors

### Error: `No such DSL method 'pipeline'`
**Cause:** Using Declarative Pipeline syntax without the Pipeline plugin.
**Fix:** Install "Pipeline" plugin from Jenkins plugin manager, or use Scripted Pipeline syntax.

### Error: `Scripts not permitted to use method`
**Cause:** Script Security plugin blocking unapproved methods.
**Fix:**
1. Go to Manage Jenkins → In-process Script Approval
2. Approve the pending script

### Error: `java.io.NotSerializableException`
**Cause:** Non-serializable objects used in pipeline.
**Fix:** Use `@NonCPS` annotation or restructure code.

## Jenkinsfile Basics

```groovy
pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                sh 'npm install'
                sh 'npm run build'
            }
        }
        stage('Test') {
            steps {
                sh 'npm test'
            }
        }
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh 'kubectl apply -f k8s/'
            }
        }
    }

    post {
        failure {
            echo 'Pipeline failed!'
        }
        success {
            echo 'Pipeline succeeded!'
        }
    }
}
```

## Agent Issues

### Build Stuck in Queue
**Causes:**
1. No agents available
2. Agent offline
3. Label mismatch

```bash
# Check: Manage Jenkins → Nodes and Clouds
# Verify agents are connected and have matching labels
```

### Agent Disconnected
**Fix:**
1. Check network connectivity to agent
2. Restart the agent service
3. Check agent logs at `/var/log/jenkins/`

## Credentials Issues

### Error: `Could not find credentials`
```groovy
// Use credentials binding
pipeline {
    environment {
        DOCKER_CREDS = credentials('docker-hub-credentials')
    }
    stages {
        stage('Push') {
            steps {
                sh 'docker login -u $DOCKER_CREDS_USR -p $DOCKER_CREDS_PSW'
            }
        }
    }
}
```

## Workspace Issues

### Disk Space Full
```bash
# Clean old builds
# Manage Jenkins → System Configuration → Discard Old Builds

# Clean workspace in pipeline
post {
    always {
        cleanWs()
    }
}
```
