import os

package_version = '1.10.0'
with_gpu = True 
platform = 'linux'
# platform = 'windows'
# platform = 'mac'

base_uri = 'https://github.com/microsoft/onnxruntime/releases/download'
full_uri = ''
archive_name = ''
archive_extension = ''

if platform.lower() == 'windows':
    package_name = 'onnxruntime-win-x64'
    if with_gpu:
        package_name += '-gpu'
    
    archive_extension = 'zip'
    archive_name = f'{package_name}-{package_version}'
    full_uri   = f'{base_uri}/v{package_version}/{archive_name}.{archive_extension}'

elif platform.lower() == 'linux':
    package_name = 'onnxruntime-linux-x64'
    if with_gpu:
        package_name += '-gpu'
    
    archive_extension = 'tgz'
    archive_name = f'{package_name}-{package_version}'
    full_uri = f'{base_uri}/v{package_version}/{archive_name}.{archive_extension}'

elif platform.lower() == 'mac':
    package_name = 'onnxruntime-osx-universal2'
    if with_gpu:
        print("GPU support not available")
    
    archive_extension = 'tgz'
    archive_name = f'{package_name}-{package_version}'
    full_uri = f'{base_uri}/v{package_version}/{archive_name}.{archive_extension}'

pwd = os.path.abspath(os.getcwd())
print(f'Downloading onnxruntime binaries in: {pwd}')

print(full_uri)

def download():
    import requests
    file_path = os.path.join(pwd, f'onnxruntime.{archive_extension}')
    with open(file_path, "wb") as file:
        response = requests.get(full_uri)
        file.write(response.content)

def unzip():
    import shutil
    file_path = os.path.join(pwd, f'onnxruntime.{archive_extension}')
    extract_dir = os.path.join(pwd, 'onnxruntime')
    shutil.unpack_archive(file_path, extract_dir)
    os.remove(file_path)

def rename():
    src_dir = os.path.join(pwd, 'onnxruntime', archive_name)
    dst_dir = os.path.join(pwd, 'onnxruntime', platform)
    os.rename(src_dir, dst_dir)

download()
unzip()
rename()