import os
import torch
from datetime import datetime
from functools import wraps
from time import time
import torch
from glob import glob
import random
import numpy as np

def currtime_str() -> str:
  return datetime.now().strftime("%m-%d_%H-%M")

def save_model_currtime(model_sd, modelname: str, save_folder: str = 'trained') -> None:
  if save_folder:
    pth = f'{save_folder}/{modelname}_'+ datetime.now().strftime("%m-%d_%H-%M") + '.pth'
    os.makedirs(f'./{save_folder}', exist_ok=True)
  else:
    pth = f'{modelname}_'+ datetime.now().strftime("%m-%d_%H-%M") + '.pth'
  
  torch.save(model_sd, pth)

def save_model(model_sd, trained:bool = True, save_folder: str='', model_name: str=''):
  folder_pth = os.path.join('.', 'trained') if trained else '.'
  if save_folder:
    folder_pth = os.path.join(folder_pth, save_folder)
  os.makedirs(folder_pth, exist_ok=True)
  pth = os.path.join(folder_pth, model_name)
  torch.save(model_sd, f'{pth}.pth')

def print_model_pth(save_folder=''):
  path = f'./{save_folder}' if save_folder else ''

  #files = [f for f in os.listdir(path) if f.endswith('.pth')]
  files = [f for f in sorted(glob(f'{path}/*.pth'))]

  if not files:
    raise FileNotFoundError('No .pth files found.')

  for i, f in enumerate(files):
    print(f'[{i}] {f}')

def select_and_torch_load_from(folder=''):
  '''
  returns torch.load(full_path)

  usage: 
  model.load_state_dict(base.select_and_torch_load_from(folder='trained'))
  '''
  path = f'./{folder}' if folder else ''

  #files = [f for f in os.listdir(path) if f.endswith('.pth')]
  files = [f for f in sorted(glob(f'{path}/*.pth'))]

  if not files:
    raise FileNotFoundError('No .pth files found.')
  
  #for i, f in enumerate(files):
  #  print(f'[{i}] {f}')
  
  idx = int(input('Select file number : '))
  if idx < 0 or idx >= len(files):
    raise IndexError('Invalid selection.')
  
  full_path = files[idx] # os.path.join(path, files[idx])

  return torch.load(full_path)

def get_model_pths(save_folder='', reverse: bool=False):
  path = f'./{save_folder}' if save_folder else ''

  #files = [f for f in os.listdir(path) if f.endswith('.pth')]
  files = [f for f in sorted(glob(f'{path}/*.pth'), key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=reverse)]

  if not files:
    raise FileNotFoundError('No .pth files found.')

  return files

def set_and_torch_load_from(folder='', idx=0):
  files = get_model_pths(folder)
  if idx < 0 or idx > len(files):
    raise IndexError('Invalid selection.')
  
  return torch.load(files[idx])
  
def mesure_time(func):
  @wraps(func)
  def wrapper(*args, **kargs):
    t1 = time()
    func(*args, **kargs)
    print(f'[총 실행 시간: {time()-t1:.1f}초]')
    return func
  return wrapper

def set_cuda_seed_print_device(device, seed):
  '''
  워커마다 시드 고정, 생성기 고정까지 하면 (DataLoader 재현성) 재현 가능
  '''
  if device == 'cuda':
    print('[using cuda]')
  else:
    print('[using cpu]')
  torch.manual_seed(seed)
  if device == 'cuda':
    torch.cuda.manual_seed_all(seed)
  random.seed(seed) 
  np.random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  
def get_version_from(save_folder: str, trained: bool=True) -> int:
  """
  use for config["save_folder"], config["run_name"]
  ex)
  config["save_folder"] = os.path.join(project_name, f"v{ver}")
  config["run_name"] = f"(onecycle_align_lr_{ver}:{config['lr']})"
  """
  folder_pth = os.path.join('.', 'trained') if trained else '.'
  if save_folder:
    folder_pth = os.path.join(folder_pth, save_folder)
  os.makedirs(folder_pth, exist_ok=True)
  version_path = os.path.join(folder_pth, "version.txt")
  if not os.path.exists(version_path):
    with open(version_path, "wt") as f:
      f.write(str(1))
      return 1
  else:
    with open(version_path, "r") as f:
      ver = int(f.read().strip())
      return ver
    
def add_version_from(save_folder: str, trained: bool=True):
  """
  raises FileNotFoundError if ./(trained/)save_folder/version.txt not exist
  """
  folder_pth = os.path.join('.', 'trained') if trained else '.'
  if save_folder:
    folder_pth = os.path.join(folder_pth, save_folder)
  version_path = os.path.join(folder_pth, "version.txt")
  if not os.path.exists(version_path):
    raise FileNotFoundError
  else:
    with open(version_path, "r") as f:
      ver = int(f.read().strip())
    ver += 1
    with open(version_path, "w") as f:
      f.write(str(ver))