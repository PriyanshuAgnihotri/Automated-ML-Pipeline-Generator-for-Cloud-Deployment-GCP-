import pytest
import yaml
import tempfile
import os
from src.utils import load_config, validate_config

def test_load_config():
    # Create a temporary config file
    config_data = {
        'pipeline': {'name': 'test'},
        'data_processing': {'input_path': 'test.csv'}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        config = load_config(temp_path)
        assert config['pipeline']['name'] == 'test'
        assert config['data_processing']['input_path'] == 'test.csv'
    finally:
        os.unlink(temp_path)

def test_validate_config():
    valid_config = {
        'data_processing': {},
        'training': {},
        'validation': {},
        'deployment': {}
    }
    
    invalid_config = {
        'data_processing': {},
        'training': {}
        # Missing validation and deployment
    }
    
    assert validate_config(valid_config, "pipeline") == True
    
    with pytest.raises(ValueError):
        validate_config(invalid_config, "pipeline")