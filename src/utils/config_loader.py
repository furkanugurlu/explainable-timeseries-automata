import os
import yaml
from pathlib import Path
from typing import Any, Dict

def get_project_root() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).parent.parent.parent

def load_config(config_name: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads a YAML configuration file.
    
    Args:
        config_name (str): The name of the configuration file in the 'configs' directory.
        
    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    root_dir = get_project_root()
    config_path = root_dir / "configs" / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML configuration: {e}")

if __name__ == "__main__":
    # Example usage
    try:
        cfg = load_config()
        import pprint
        print("Loaded Configuration Successfully:")
        pprint.pprint(cfg)
    except Exception as error:
        print(f"Failed to load config: {error}")
