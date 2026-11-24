# build_any.py module func shapes_json  (run with NKI_ARTIFACTS_DIR set): calls module.func[2](*zeros)
import sys, json, importlib, numpy as np, ml_dtypes
mod, fn, spec = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
dt = {"bf16": ml_dtypes.bfloat16, "f32": np.float32, "i32": np.int32}
args = [np.zeros(tuple(s), dt[d]) for s, d in spec]
getattr(importlib.import_module(mod), fn)[2](*args)
