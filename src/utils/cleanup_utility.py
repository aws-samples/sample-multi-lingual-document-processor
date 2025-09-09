"""
Cleanup Utility for Multi-lingual Document Processor
Handles cleanup of temporary files and directories for both Lambda and SageMaker environments.
"""

import os
import shutil
import logging
import glob
from pathlib import Path
from src.config.config import TEMP_DIRECTORIES

class CleanupUtility:
    """
    Utility class for cleaning up temporary files and directories.
    """
    
    def __init__(self):
        """
        Initialize the cleanup utility.
        """
        self.logger = logging.getLogger(__name__)
        self.temp_directories = TEMP_DIRECTORIES
        
        # Additional common temporary locations
        self.additional_temp_locations = [
            "/tmp/combined_output.txt",
            "/tmp/*.pdf",
            "/tmp/*.txt",
            "/tmp/*.json",
            "/tmp/*.csv",
            "/tmp/*.png",
            "/tmp/*.jpg",
            "/tmp/*.jpeg"
        ]
    
    def cleanup_temp_directories(self):
        """
        Clean up all temporary directories and their contents recursively.
        """
        self.logger.info("Starting cleanup of temporary directories")
        
        for directory in self.temp_directories:
            try:
                if os.path.exists(directory):
                    self.logger.info(f"Cleaning directory: {directory}")
                    
                    # Use shutil.rmtree for recursive removal
                    shutil.rmtree(directory, ignore_errors=False)
                    self.logger.info(f"Successfully removed directory: {directory}")
                else:
                    self.logger.debug(f"Directory does not exist: {directory}")
                    
            except Exception as e:
                self.logger.warning(f"Error removing directory {directory}: {str(e)}")
                # Try to clean individual files if directory removal fails
                self._cleanup_directory_contents(directory)
    
    def cleanup_temp_files(self):
        """
        Clean up individual temporary files using glob patterns.
        """
        self.logger.info("Starting cleanup of temporary files")
        
        for pattern in self.additional_temp_locations:
            try:
                matching_files = glob.glob(pattern)
                for file_path in matching_files:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        self.logger.info(f"Removed file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path, ignore_errors=True)
                        self.logger.info(f"Removed directory: {file_path}")
                        
            except Exception as e:
                self.logger.warning(f"Error cleaning files with pattern {pattern}: {str(e)}")
    
    def _cleanup_directory_contents(self, directory):
        """
        Fallback method to clean directory contents when rmtree fails.
        
        Args:
            directory (str): Directory path to clean
        """
        try:
            if not os.path.exists(directory):
                return
                
            for root, dirs, files in os.walk(directory, topdown=False):
                # Remove all files
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        self.logger.debug(f"Removed file: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"Error removing file {file_path}: {str(e)}")
                
                # Remove all directories
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        os.rmdir(dir_path)
                        self.logger.debug(f"Removed directory: {dir_path}")
                    except Exception as e:
                        self.logger.warning(f"Error removing directory {dir_path}: {str(e)}")
            
            # Finally remove the root directory
            try:
                os.rmdir(directory)
                self.logger.info(f"Removed root directory: {directory}")
            except Exception as e:
                self.logger.warning(f"Error removing root directory {directory}: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error in fallback cleanup for {directory}: {str(e)}")
    
    def cleanup_lambda_temp_files(self):
        """
        Specific cleanup for Lambda environment temporary files.
        """
        self.logger.info("Starting Lambda-specific cleanup")
        
        lambda_temp_patterns = [
            "/tmp/*",
            "/var/task/tmp/*" if os.path.exists("/var/task/tmp") else None
        ]
        
        for pattern in lambda_temp_patterns:
            if pattern is None:
                continue
                
            try:
                matching_items = glob.glob(pattern)
                for item_path in matching_items:
                    # Skip system directories and files
                    if any(skip in item_path for skip in ['/tmp/systemd', '/tmp/.X', '/tmp/.ICE']):
                        continue
                        
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        self.logger.debug(f"Removed Lambda temp file: {item_path}")
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                        self.logger.debug(f"Removed Lambda temp directory: {item_path}")
                        
            except Exception as e:
                self.logger.warning(f"Error in Lambda cleanup with pattern {pattern}: {str(e)}")
    
    def cleanup_sagemaker_temp_files(self):
        """
        Specific cleanup for SageMaker environment temporary files.
        """
        self.logger.info("Starting SageMaker-specific cleanup")
        
        sagemaker_temp_patterns = [
            "/opt/ml/processing/temp/*",
            "/opt/ml/processing/output/temp/*",
            "/tmp/*"
        ]
        
        for pattern in sagemaker_temp_patterns:
            try:
                matching_items = glob.glob(pattern)
                for item_path in matching_items:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        self.logger.debug(f"Removed SageMaker temp file: {item_path}")
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                        self.logger.debug(f"Removed SageMaker temp directory: {item_path}")
                        
            except Exception as e:
                self.logger.warning(f"Error in SageMaker cleanup with pattern {pattern}: {str(e)}")
    
    def full_cleanup(self, environment="auto"):
        """
        Perform complete cleanup of all temporary files and directories.
        
        Args:
            environment (str): Environment type - "lambda", "sagemaker", or "auto"
        """
        self.logger.info(f"Starting full cleanup for environment: {environment}")
        
        try:
            # Clean up configured temp directories
            self.cleanup_temp_directories()
            
            # Clean up additional temp files
            self.cleanup_temp_files()
            
            # Environment-specific cleanup
            if environment == "lambda":
                self.cleanup_lambda_temp_files()
            elif environment == "sagemaker":
                self.cleanup_sagemaker_temp_files()
            elif environment == "auto":
                # Auto-detect environment
                if os.path.exists("/opt/ml/processing"):
                    self.cleanup_sagemaker_temp_files()
                elif os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
                    self.cleanup_lambda_temp_files()
                else:
                    # Default to both
                    self.cleanup_lambda_temp_files()
                    self.cleanup_sagemaker_temp_files()
            
            self.logger.info("Full cleanup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during full cleanup: {str(e)}")
    
    def get_temp_directory_sizes(self):
        """
        Get sizes of temporary directories for monitoring.
        
        Returns:
            dict: Dictionary with directory paths and their sizes in bytes
        """
        directory_sizes = {}
        
        for directory in self.temp_directories:
            try:
                if os.path.exists(directory):
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(directory):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            try:
                                total_size += os.path.getsize(filepath)
                            except (OSError, IOError):
                                pass
                    directory_sizes[directory] = total_size
                else:
                    directory_sizes[directory] = 0
                    
            except Exception as e:
                self.logger.warning(f"Error calculating size for {directory}: {str(e)}")
                directory_sizes[directory] = -1
        
        return directory_sizes


def cleanup_for_lambda():
    """
    Convenience function for Lambda cleanup.
    """
    cleanup_util = CleanupUtility()
    cleanup_util.full_cleanup(environment="lambda")


def cleanup_for_sagemaker():
    """
    Convenience function for SageMaker cleanup.
    """
    cleanup_util = CleanupUtility()
    cleanup_util.full_cleanup(environment="sagemaker")


def cleanup_auto():
    """
    Convenience function for auto-detected environment cleanup.
    """
    cleanup_util = CleanupUtility()
    cleanup_util.full_cleanup(environment="auto")


if __name__ == "__main__":
    # For testing purposes
    logging.basicConfig(level=logging.INFO)
    cleanup_auto()