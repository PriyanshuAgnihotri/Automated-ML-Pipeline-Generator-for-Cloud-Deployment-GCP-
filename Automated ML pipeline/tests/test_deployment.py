import pytest
from unittest.mock import Mock, patch
from src.deployment import deploy_to_ai_platform, create_cloud_function_trigger

@patch('src.deployment.aiplatform')
def test_deploy_to_ai_platform(mock_aiplatform):
    # Mock AI Platform components
    mock_model = Mock()
    mock_endpoint = Mock()
    mock_endpoint.resource_name = "projects/test/locations/us-central1/endpoints/123"
    
    mock_aiplatform.Model.upload.return_value = mock_model
    mock_model.deploy.return_value = mock_endpoint
    
    # Test config
    config = {
        'deployment': {
            'project_id': 'test-project',
            'region': 'us-central1',
            'serving_container_image': 'gcr.io/test/image',
            'machine_type': 'n1-standard-2',
            'min_replica_count': 1,
            'max_replica_count': 1
        },
        'pipeline': {
            'name': 'test-pipeline'
        }
    }
    
    # Test deployment
    endpoint = deploy_to_ai_platform('gs://test-bucket/model.joblib', config)
    
    assert endpoint.resource_name == "projects/test/locations/us-central1/endpoints/123"
    mock_aiplatform.init.assert_called_once_with(project='test-project', location='us-central1')
    mock_aiplatform.Model.upload.assert_called_once()
    mock_model.deploy.assert_called_once()

@patch('src.deployment.storage')
def test_create_cloud_function_trigger(mock_storage):
    # Mock storage components
    mock_bucket = Mock()
    mock_storage.Client.return_value.bucket.return_value = mock_bucket
    
    # Test trigger creation
    result = create_cloud_function_trigger('test-bucket', 'https://us-central1-test.cloudfunctions.net/function')
    
    assert result is True
    mock_storage.Client.assert_called_once()