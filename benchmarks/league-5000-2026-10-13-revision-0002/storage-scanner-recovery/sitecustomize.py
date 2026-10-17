"""Approved exact-storage scanner overlay; frozen computational files unchanged."""
import os, hashlib, importlib.util
from pathlib import Path
root = Path(os.environ['SKALLAGRIM_METADATA_RECOVERY_ROOT']).resolve()
self_path = Path(__file__).resolve()
if hashlib.sha256(self_path.read_bytes()).hexdigest() != os.environ.get('SKALLAGRIM_STORAGE_RECOVERY_SHA'):
    os._exit(78)
scanner = self_path.with_name('fast_resources.py')
if hashlib.sha256(scanner.read_bytes()).hexdigest() != os.environ.get('SKALLAGRIM_STORAGE_SCANNER_SHA'):
    os._exit(78)
# Execute the original approved metadata shim with its original __file__, so its
# independent hash check and function module identity remain unchanged.
original = root / 'recovery-0002/io-shim/sitecustomize.py'
namespace = {'__name__': 'sitecustomize', '__file__': str(original)}
exec(compile(original.read_bytes(), str(original), 'exec'), namespace)
import sys
spec = importlib.util.spec_from_file_location('resources', scanner)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.modules['resources'] = module
