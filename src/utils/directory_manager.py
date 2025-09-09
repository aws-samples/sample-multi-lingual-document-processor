"""
Directory Manager class for handling temporary directories and file operations.
"""

import os
import logging
from pathlib import Path
from src.config.config import TEMP_DIRECTORIES

class DirectoryManager:
    """
    Manages the creation and cleanup of temporary directories.
    """
    
    def __init__(self):
        """
        Initialize the directory manager.
        """
        self.logger = logging.getLogger()
        self.temp_directories = TEMP_DIRECTORIES
    
    def create_temp_directories(self):
        """
        Create temporary directories for processing.
        """
        for directory in self.temp_directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                self.logger.info(f"Created directory: {directory}")
    
    def cleanup_temp_files(self):
        """
        Clean up temporary files and directories.
        """
        for directory in self.temp_directories:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            os.rmdir(file_path)
                    except Exception as e:
                        self.logger.warning(f"Error cleaning up {file_path}: {str(e)}")
    
    def get_files_in_directory(self, directory, pattern="*"):
        """
        Get files in a directory matching a pattern.
        
        Args:
            directory (str): Directory path
            pattern (str): File pattern to match
            
        Returns:
            list: List of file paths
        """
        path = Path(directory)
        return list(path.glob(pattern))
    
    def natural_sort_key(self, s):
        """
        Natural sort key for sorting filenames with numbers.
        
        Args:
            s: String to sort
            
        Returns:
            list: List of components for natural sorting
        """
        import re
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', s)]
