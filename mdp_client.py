#!/usr/bin/env python3
"""
Multi-lingual Document Processor (MDP) Client

This client submits processing jobs for existing files in S3 via API Gateway.
For file uploads, use a separate upload utility.
"""

import boto3
import requests
import json
import os
import sys
import argparse
import time
from typing import Optional, Dict, Any, List

class MDPClient:
    def __init__(self, stack_name: str = None, bucket_name: str = None, region: str = 'us-east-1'):
        """
        Initialize MDP Client
        
        Args:
            stack_name: CloudFormation stack name (optional if bucket_name provided)
            bucket_name: S3 bucket name (optional if stack_name provided)
            region: AWS region
        """
        self.stack_name = stack_name
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        
        if stack_name:
            self.cf_client = boto3.client('cloudformation', region_name=region)
            # Get stack outputs
            self._load_stack_info()
        elif bucket_name:
            # Direct bucket mode - no API Gateway
            self.bucket_name = bucket_name
            self.api_endpoint = None
            self.api_key_value = None
            print(f"✅ Direct S3 mode - Bucket: {bucket_name}")
        else:
            raise ValueError("Either stack_name or bucket_name must be provided")
    
    def _load_stack_info(self):
        """Load stack information from CloudFormation outputs"""
        try:
            response = self.cf_client.describe_stacks(StackName=self.stack_name)
            outputs = response['Stacks'][0]['Outputs']
            
            self.stack_info = {}
            for output in outputs:
                self.stack_info[output['OutputKey']] = output['OutputValue']
            
            # Extract key information
            self.bucket_name = self.stack_info.get('BucketName')
            self.api_endpoint = self.stack_info.get('ApiEndpoint')
            self.api_key_id = self.stack_info.get('ApiKeyId')
            self.lambda_function_name = self.stack_info.get('LambdaFunctionName')
            
            # Fix malformed API endpoint if needed
            if self.api_endpoint and ('.execute-api..amazonaws.com' in self.api_endpoint or 
                                    self.api_endpoint.count('.') < 4 or 
                                    'https://.execute-api.' in self.api_endpoint):
                print(f"🔧 Detected malformed API endpoint: {self.api_endpoint}")
                # Get the actual API Gateway ID from CloudFormation resources
                try:
                    resources = self.cf_client.describe_stack_resources(StackName=self.stack_name)
                    api_gateway_id = None
                    for resource in resources['StackResources']:
                        if resource['ResourceType'] == 'AWS::ApiGateway::RestApi':
                            api_gateway_id = resource['PhysicalResourceId']
                            break
                    
                    if api_gateway_id:
                        self.api_endpoint = f'https://{api_gateway_id}.execute-api.{self.region}.amazonaws.com/prod/process'
                        print(f"🔧 Fixed malformed API endpoint: {self.api_endpoint}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not fix API endpoint: {e}")
            
            if not all([self.bucket_name, self.api_endpoint, self.api_key_id]):
                raise ValueError("Missing required stack outputs")
            
            # Get API key value
            self._get_api_key_value()
            
            print(f"✅ Stack info loaded:")
            print(f"   Bucket: {self.bucket_name}")
            print(f"   API Endpoint: {self.api_endpoint}")
            print(f"   Lambda Function: {self.lambda_function_name}")
            
        except Exception as e:
            raise Exception(f"Failed to load stack info: {e}")
    
    def _get_api_key_value(self):
        """Get the actual API key value"""
        try:
            apigateway = boto3.client('apigateway', region_name=self.region)
            response = apigateway.get_api_key(apiKey=self.api_key_id, includeValue=True)
            self.api_key_value = response['value']
            print(f"✅ API key retrieved")
        except Exception as e:
            raise Exception(f"Failed to get API key value: {e}")
    
    def list_files_in_folder(self, s3_folder: str) -> List[Dict[str, Any]]:
        """
        List files in a specific S3 folder
        
        Args:
            s3_folder: S3 folder path
            
        Returns:
            List of file information dictionaries
        """
        try:
            # Ensure folder path ends with /
            if not s3_folder.endswith('/'):
                s3_folder += '/'
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=s3_folder
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Skip folder markers (keys ending with /)
                    if not obj['Key'].endswith('/'):
                        files.append({
                            'key': obj['Key'],
                            'filename': obj['Key'].split('/')[-1],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat(),
                            'extension': obj['Key'].split('.')[-1].lower() if '.' in obj['Key'] else 'unknown'
                        })
            
            return files
            
        except Exception as e:
            raise Exception(f"Failed to list files in folder {s3_folder}: {e}")
    
    def submit_processing_job(self, 
                            file_key: str,
                            doc_category: str = "general",
                            doc_language: str = "english",
                            output_language: str = "english",
                            bucket_name: str = None) -> Dict[Any, Any]:
        """
        Submit processing job via API Gateway or direct Lambda invocation
        
        Args:
            file_key: S3 folder path where PDF documents are located (without bucket name)
            doc_category: Document category (default: "general")
            doc_language: Document language (default: "english")
            output_language: Output language (default: "english")
            bucket_name: Override bucket name (optional)
            
        Returns:
            Processing response
        """
        try:
            # Use provided bucket or default
            target_bucket = bucket_name or self.bucket_name
            
            # Prepare payload based on lambda function expectations
            payload = {
                "file_key": file_key,
                "bucket": target_bucket,
                "doc_category": doc_category,
                "doc_language": doc_language,
                "output_language": output_language
            }
            
            print(f"🚀 Submitting processing job:")
            print(f"   S3 Folder: s3://{target_bucket}/{file_key}")
            print(f"   Category: {doc_category}")
            print(f"   Language: {doc_language}")
            print(f"   Output Language: {output_language}")
            
            # Check if we have API Gateway or direct Lambda mode
            if self.api_endpoint and self.api_key_value:
                # API Gateway mode
                headers = {
                    'Content-Type': 'application/json',
                    'x-api-key': self.api_key_value
                }
                
                response = requests.post(
                    self.api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                print(f"📡 API Response Status: {response.status_code}")
                
                if response.status_code == 202:
                    print("✅ Processing job submitted successfully")
                    return {
                        'success': True,
                        'method': 'api_gateway',
                        'status_code': response.status_code,
                        'response': response.json() if response.content else {},
                        'payload_sent': payload
                    }
                else:
                    print(f"❌ API call failed with status {response.status_code}")
                    print(f"Response: {response.text}")
                    return {
                        'success': False,
                        'method': 'api_gateway',
                        'status_code': response.status_code,
                        'response': response.text,
                        'payload_sent': payload
                    }
            else:
                # Direct mode - just validate and return payload
                print("✅ Payload prepared for direct processing")
                return {
                    'success': True,
                    'method': 'direct',
                    'message': 'Payload prepared - use with direct Lambda invocation or other processing method',
                    'payload_sent': payload
                }
                
        except Exception as e:
            print(f"❌ Failed to submit processing job: {e}")
            return {
                'success': False,
                'error': str(e),
                'payload_sent': payload if 'payload' in locals() else None
            }
    
    def process_folder(self, 
                      s3_folder: str,
                      doc_category: str = "general",
                      doc_language: str = "english",
                      output_language: str = "english",
                      file_extensions: List[str] = None) -> Dict[Any, Any]:
        """
        Process all files in an S3 folder
        
        Args:
            s3_folder: S3 folder path where PDF documents are located
            doc_category: Document category
            doc_language: Document language
            output_language: Output language
            file_extensions: List of file extensions to process (e.g., ['pdf', 'txt'])
            
        Returns:
            Processing result with file information
        """
        try:
            # List files in folder
            files = self.list_files_in_folder(s3_folder)
            
            # Filter by extensions if specified
            if file_extensions:
                file_extensions = [ext.lower().lstrip('.') for ext in file_extensions]
                files = [f for f in files if f['extension'] in file_extensions]
            
            if not files:
                return {
                    'success': False,
                    'error': f'No files found in folder {s3_folder}' + 
                            (f' with extensions {file_extensions}' if file_extensions else ''),
                    'files_found': []
                }
            
            print(f"📁 Found {len(files)} files in {s3_folder}:")
            for file_info in files:
                print(f"   📄 {file_info['filename']} ({file_info['size']} bytes)")
            
            # Submit processing job for the folder
            result = self.submit_processing_job(s3_folder, doc_category, doc_language, output_language)
            
            # Add file information to result
            result['folder_info'] = {
                'folder_path': s3_folder,
                'files_found': files,
                'file_count': len(files),
                'total_size': sum(f['size'] for f in files)
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def check_processing_status(self) -> Dict[Any, Any]:
        """
        Check processing status by looking at S3 processed folder
        """
        try:
            # List objects in processed folder
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='processed/',
                MaxKeys=10
            )
            
            if 'Contents' in response:
                files = [obj['Key'] for obj in response['Contents']]
                return {
                    'success': True,
                    'processed_files': files,
                    'count': len(files)
                }
            else:
                return {
                    'success': True,
                    'processed_files': [],
                    'count': 0,
                    'message': 'No processed files found yet'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_folder_contents(self, folder_path: str = "uploads") -> Dict[Any, Any]:
        """List files in specified folder"""
        try:
            files = self.list_files_in_folder(folder_path)
            
            return {
                'success': True,
                'folder': folder_path,
                'files': files,
                'count': len(files),
                'total_size': sum(f['size'] for f in files) if files else 0
            }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

def main():
    parser = argparse.ArgumentParser(description='MDP Client - Process documents in S3')
    parser.add_argument('--stack-name', '-s', help='CloudFormation stack name')
    parser.add_argument('--bucket', '-b', help='S3 bucket name (alternative to stack-name)')
    parser.add_argument('--category', '-c', default='report', help='Document category')
    parser.add_argument('--language', '-l', default='english', help='Document language')
    parser.add_argument('--output-language', '-o', default='english', help='Output language')
    parser.add_argument('--folder', '-f', default='uploads', help='S3 folder to process (where PDFs are located)')
    parser.add_argument('--region', '-r', default='us-east-1', help='AWS region')
    parser.add_argument('--list-folder', help='List files in specified folder')
    parser.add_argument('--check-status', action='store_true', help='Check processing status')
    parser.add_argument('--process-folder', help='Process all files in specified folder')
    parser.add_argument('--extensions', nargs='+', help='File extensions to process (e.g., pdf txt docx)')
    
    args = parser.parse_args()
    
    try:
        # Validate arguments
        if not args.stack_name and not args.bucket:
            print("❌ Error: Either --stack-name or --bucket must be provided")
            sys.exit(1)
        
        # Initialize client
        if args.stack_name:
            print(f"🔧 Initializing MDP Client for stack: {args.stack_name}")
            client = MDPClient(stack_name=args.stack_name, region=args.region)
        else:
            print(f"🔧 Initializing MDP Client for bucket: {args.bucket}")
            client = MDPClient(bucket_name=args.bucket, region=args.region)
        
        if args.list_folder:
            print(f"\n📋 Listing files in folder: {args.list_folder}")
            result = client.list_folder_contents(args.list_folder)
            if result['success']:
                if result['count'] > 0:
                    print(f"   Found {result['count']} files ({result['total_size']} bytes total):")
                    for file_info in result['files']:
                        print(f"   📄 {file_info['filename']} ({file_info['size']} bytes, .{file_info['extension']})")
                else:
                    print(f"   No files found in folder: {args.list_folder}")
            else:
                print(f"   ❌ Error: {result['error']}")
        
        elif args.check_status:
            print("\n🔍 Checking processing status:")
            result = client.check_processing_status()
            if result['success']:
                if result['count'] > 0:
                    print(f"   Found {result['count']} processed files:")
                    for file_key in result['processed_files']:
                        print(f"   📄 {file_key}")
                else:
                    print("   No processed files found yet")
            else:
                print(f"   ❌ Error: {result['error']}")
        
        elif args.process_folder:
            print(f"\n🚀 Processing folder: {args.process_folder}")
            result = client.process_folder(
                args.process_folder, 
                args.category, 
                args.language,
                args.output_language,
                args.extensions
            )
            print(f"\n📊 Result: {json.dumps(result, indent=2, default=str)}")
        
        else:
            # Default action - process the specified folder
            print(f"\n🚀 Processing default folder: {args.folder}")
            result = client.process_folder(
                args.folder,
                args.category,
                args.language,
                args.output_language,
                args.extensions
            )
            print(f"\n📊 Result: {json.dumps(result, indent=2, default=str)}")
        
        # Show usage examples
        print("\n📝 Usage Examples:")
        if args.stack_name:
            print(f"  python {sys.argv[0]} --stack-name {args.stack_name} --process-folder uploads --language japanese --output-language english")
            print(f"  python {sys.argv[0]} --stack-name {args.stack_name} --list-folder uploads")
            print(f"  python {sys.argv[0]} --stack-name {args.stack_name} --folder documents --category invoice --extensions pdf")
        else:
            print(f"  python {sys.argv[0]} --bucket my-bucket --process-folder documents --extensions pdf --output-language english")
            print(f"  python {sys.argv[0]} --bucket my-bucket --list-folder uploads")
            print(f"  python {sys.argv[0]} --bucket my-bucket --folder uploads --category general --language english")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
    def validate_api_endpoint(self):
        """Validate API endpoint format"""
        if self.api_endpoint:
            if not self.api_endpoint.startswith('https://'):
                print(f"⚠️  Warning: API endpoint should start with https://: {self.api_endpoint}")
                return False
            elif '.execute-api.' not in self.api_endpoint:
                print(f"⚠️  Warning: API endpoint should contain .execute-api.: {self.api_endpoint}")
                return False
            elif self.api_endpoint.count('.') < 4:
                print(f"⚠️  Warning: API endpoint format may be incorrect: {self.api_endpoint}")
                return False
            else:
                print(f"✅ API endpoint format validated")
                return True
        return False