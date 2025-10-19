"""
ST-GCN Tester Module
test.py와 완전히 동일한 구조로 작동하는 테스트 모듈
Runner.from_cfg() -> runner.test() 호출 -> result.pkl 파싱 및 반환
"""
import os
import os.path as osp
import pickle
import sys
import tempfile
import uuid
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner

# Do not import mmaction.registry at module import time. Import it later
# after we ensure the local /mmaction2 repo is on sys.path so the registry
# is populated from the correct package (avoids stale/site-package binding).
from modules.utils import debug_log, csv_to_pkl


def prepare_config_for_test(csv_path: Path):
    """
    test.py의 parse_args()와 merge_args() 역할을 수행
    CSV 파일을 받아서 config를 준비하고 필요한 설정을 오버라이드
    """
    # 1. Config 파일 경로 (my_stgcnpp.py)
    config_path = Path(__file__).parent / "my_stgcnpp.py"
    checkpoint_path = None
    
    # checkpoint 경로 찾기
    for p in [
        Path(__file__).parent.parent / "stgcn_70p.pth"
    ]:
        if Path(p).exists():
            checkpoint_path = str(p)
            break
    
    if checkpoint_path is None:
        raise FileNotFoundError("checkpoint file stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth not found")
    
    debug_log(f"Using config: {config_path}")
    debug_log(f"Using checkpoint: {checkpoint_path}")
    
    # 2. CSV를 ann.pkl로 변환
    unique_id = uuid.uuid4().hex[:8]
    ann_pkl_path = Path(tempfile.gettempdir()) / f"test_ann_{unique_id}.pkl"
    debug_log(f"Converting CSV to PKL: {csv_path} -> {ann_pkl_path}")
    csv_to_pkl(csv_path, ann_pkl_path)
    
    # 3. result.pkl 경로 설정
    result_pkl_path = Path(tempfile.gettempdir()) / f"test_result_{unique_id}.pkl"
    
    # Make sure local mmaction2 package is importable so model classes (e.g. RecognizerGCN)
    # are registered before loading the config. We try the repo-local path and the
    # container path '/mmaction2'.
    repo_dir = Path(__file__).parent
    candidate = repo_dir.parent / "mmaction2"
    try:
        if candidate.exists():
            sys.path.insert(0, str(candidate))
        else:
            # fallback to container path
            sys.path.insert(0, "/mmaction2")
    except Exception:
        pass

    # Ensure mmaction registers its modules (models, datasets, etc.).
    # Some mmaction registration happens when importing subpackages; call
    # register_all_modules if available to guarantee registries are populated.
    try:
        import mmaction  # noqa: F401
        try:
            # mmaction provides a helper to register all modules
            from mmaction.utils.setup_env import register_all_modules
            register_all_modules(init_default_scope=True)
            # Debug: list some registered model keys to verify GCN is present
            try:
                from mmaction.registry import MODELS
                model_keys = list(MODELS.module_dict.keys()) if hasattr(MODELS, 'module_dict') else []
                debug_log(f"Registered MODELS sample (len={len(model_keys)}): {model_keys[:50]}")
                debug_log(f"RecognizerGCN registered? {'RecognizerGCN' in model_keys}")
            except Exception as _e:
                debug_log(f"Failed to inspect MODELS registry: {_e}")
        except Exception:
            # Fallback: at least import models subpackage to trigger module imports
            try:
                import mmaction.models  # noqa: F401
                # Try to inspect MODELS registry even if register_all_modules unavailable
                try:
                    from mmaction.registry import MODELS
                    model_keys = list(MODELS.module_dict.keys()) if hasattr(MODELS, 'module_dict') else []
                    debug_log(f"Registered MODELS sample (fallback) (len={len(model_keys)}): {model_keys[:50]}")
                    debug_log(f"RecognizerGCN registered? {'RecognizerGCN' in model_keys}")
                except Exception as _e:
                    debug_log(f"Failed to inspect MODELS registry in fallback: {_e}")
            except Exception:
                # swallow; Config.fromfile() will still run and may raise a helpful error
                pass
    except Exception:
        # If mmaction cannot be imported at all, let Config.fromfile raise later
        pass

    # 4. Config 로드
    cfg = Config.fromfile(str(config_path))
    # Debug: what model type is requested in config
    try:
        cfg_model_type = cfg.model.type if hasattr(cfg, 'model') and isinstance(cfg.model, dict) and 'type' in cfg.model else getattr(cfg.model, 'type', None)
        debug_log(f"Config model.type -> {cfg_model_type}")
    except Exception as _e:
        debug_log(f"Failed to read cfg.model.type: {_e}")
    
    # 5. test.py의 merge_args와 동일한 설정 적용
    # work_dir 설정 (test.py와 동일한 우선순위)
    if cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('/tmp/work_dirs', 
                               osp.splitext(osp.basename(str(config_path)))[0])
    
    # 6. checkpoint 로드 설정
    cfg.load_from = checkpoint_path
    
    # 7. test_dataloader의 ann_file 오버라이드
    cfg.test_dataloader.dataset.ann_file = str(ann_pkl_path)
    
    # 8. DumpResults 설정 (test.py의 --dump 옵션과 동일)
    dump_metric = dict(type='DumpResults', out_file_path=str(result_pkl_path))
    if isinstance(cfg.test_evaluator, (list, tuple)):
        cfg.test_evaluator = list(cfg.test_evaluator)
        # 기존 DumpResults가 있으면 제거
        cfg.test_evaluator = [e for e in cfg.test_evaluator if e.get('type') != 'DumpResults']
        cfg.test_evaluator.append(dump_metric)
    else:
        cfg.test_evaluator = [cfg.test_evaluator, dump_metric]
    
    # 9. launcher 설정
    cfg.launcher = 'none'
    
    # 10. 환경 설정 (안정성을 위해)
    if hasattr(cfg, 'env_cfg'):
        if hasattr(cfg.env_cfg, 'mp_cfg'):
            cfg.env_cfg.mp_cfg.mp_start_method = 'fork'
        if hasattr(cfg.env_cfg, 'dist_cfg'):
            cfg.env_cfg.dist_cfg.backend = 'gloo'
    
    # 11. visualization 비활성화 (서버 환경에서 불필요)
    if hasattr(cfg, 'default_hooks') and isinstance(cfg.default_hooks, dict):
        if 'visualization' in cfg.default_hooks:
            cfg.default_hooks.visualization.enable = False
    
    debug_log(f"Config prepared: work_dir={cfg.work_dir}")
    debug_log(f"Result will be saved to: {result_pkl_path}")
    
    return cfg, ann_pkl_path, result_pkl_path


def run_stgcn_test(csv_path: Path):
    """
    test.py의 main() 함수와 완전히 동일한 구조
    1. Config 로드 및 설정
    2. Runner 생성
    3. runner.test() 실행
    4. result.pkl 파싱 및 반환
    """
    debug_log(f"run_stgcn_test start: {csv_path}")

    ann_pkl_path = None
    result_pkl_path = None

    try:
        # 1) Prepare config and temporary files
        cfg, ann_pkl_path, result_pkl_path = prepare_config_for_test(csv_path)

        # 2) Choose execution mode: inline (default) or subprocess when explicitly requested
        use_subproc = os.environ.get('MMACTION_USE_SUBPROCESS', '0') == '1'
        if use_subproc:
            debug_log("MMACTION_USE_SUBPROCESS=1 -> running stgcn_subproc in subprocess")
            import subprocess, tempfile

            subproc_script = Path(__file__).parent / "stgcn_subproc.py"
            env = os.environ.copy()
            repo_dir = Path(__file__).parent.parent
            candidate = repo_dir / "mmaction2"
            env_pythonpath = str(candidate) if candidate.exists() else "/mmaction2"
            if env.get('PYTHONPATH'):
                env['PYTHONPATH'] = env_pythonpath + os.pathsep + env['PYTHONPATH']
            else:
                env['PYTHONPATH'] = env_pythonpath

            cmd = [
                sys.executable,
                str(subproc_script),
                '--config', str(Path(__file__).parent / 'my_stgcnpp.py'),
                '--checkpoint', str(cfg.load_from),
                '--ann', str(ann_pkl_path),
                '--out', str(result_pkl_path),
            ]

            cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES')
            if cuda_env:
                env['CUDA_VISIBLE_DEVICES'] = cuda_env
                debug_log(f"Forwarding CUDA_VISIBLE_DEVICES={cuda_env} to subprocess")
            mma_device = os.environ.get('MMACTION_DEVICE')
            if mma_device:
                cmd += ['--device', mma_device]
                debug_log(f"Passing device='{mma_device}' to subprocess")

            debug_log(f"Running subprocess: {cmd} with PYTHONPATH={env['PYTHONPATH']}")
            out_log = Path(tempfile.gettempdir()) / f"stgcn_subproc_{uuid.uuid4().hex[:8]}.stdout.log"
            err_log = Path(tempfile.gettempdir()) / f"stgcn_subproc_{uuid.uuid4().hex[:8]}.stderr.log"
            debug_log(f"Subprocess stdout redirected to: {out_log}")
            debug_log(f"Subprocess stderr redirected to: {err_log}")

            with open(out_log, 'wb') as _outf, open(err_log, 'wb') as _errf:
                proc = subprocess.Popen(cmd, stdout=_outf, stderr=_errf, env=env)
                try:
                    proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    debug_log("Subprocess timed out and was killed")
                    try:
                        debug_log(f"subproc stdout (partial):\n{out_log.read_text(errors='replace')}")
                        debug_log(f"subproc stderr (partial):\n{err_log.read_text(errors='replace')}")
                    except Exception:
                        pass
                    raise RuntimeError("stgcn_subproc timed out")

            try:
                out_text = out_log.read_text(errors='replace')
                err_text = err_log.read_text(errors='replace')
                max_len = 10000
                debug_log(f"subproc stdout (tail):\n{out_text[-max_len:]}")
                debug_log(f"subproc stderr (tail):\n{err_text[-max_len:]}")
            except Exception as _e:
                debug_log(f"Failed to read subproc logs: {_e}")

            if proc.returncode != 0:
                raise RuntimeError(f"stgcn_subproc failed (exit {proc.returncode}); see logs: {err_log}")

            debug_log("Subprocess completed successfully")
            if not result_pkl_path.exists():
                raise FileNotFoundError(f"Result file not found: {result_pkl_path}")
            debug_log(f"Loading result from: {result_pkl_path}")
            with open(result_pkl_path, "rb") as f:
                result_data = pickle.load(f)
            parsed_result = parse_test_result(result_data)
            debug_log(f"Test completed successfully: {parsed_result}")
            return parsed_result

        # Inline execution (default)
        debug_log("Running ST-GCN test inline (same process)")

        # Ensure local repo on sys.path
        try:
            repo_dir = Path(__file__).parent.parent
            candidate = repo_dir / "mmaction2"
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            elif "/mmaction2" not in sys.path:
                sys.path.insert(0, "/mmaction2")
        except Exception:
            pass

        # Register modules and sync registries
        try:
            try:
                from mmaction.utils import register_all_modules
            except Exception:
                try:
                    from mmaction.utils.setup_env import register_all_modules
                except Exception:
                    register_all_modules = None
            if register_all_modules:
                register_all_modules(init_default_scope=True)
        except Exception:
            debug_log("Warning: register_all_modules failed in inline mode")

        try:
            # Explicitly import dataset modules that perform registration as a side-effect.
            try:
                import mmaction.datasets.pose_dataset  # noqa: F401
                debug_log("Imported mmaction.datasets.pose_dataset in inline mode")
            except Exception as _e:
                debug_log(f"Failed to import mmaction.datasets.pose_dataset: {_e}")

            import importlib
            import mmengine.registry as _me_reg
            mma_reg = importlib.import_module('mmaction.registry')
            # Sync additional registries including METRICS and EVALUATOR which
            # are needed to build evaluators like AccMetric.
            for reg_name in ('MODELS', 'DATASETS', 'TRANSFORMS', 'METRICS', 'EVALUATOR'):
                mma_reg_obj = getattr(mma_reg, reg_name, None)
                me_reg_obj = getattr(_me_reg, reg_name, None)
                if mma_reg_obj is None or me_reg_obj is None:
                    debug_log(f"Registry {reg_name} missing in mmaction or mmengine; skipping")
                    continue
                mma_dict = getattr(mma_reg_obj, 'module_dict', {}) or {}
                me_dict = getattr(me_reg_obj, 'module_dict', {}) or {}
                added = 0
                for name, cls in mma_dict.items():
                    if name not in me_dict:
                        try:
                            me_reg_obj.register_module(module=cls, name=name, force=True)
                            added += 1
                        except Exception as _e:
                            debug_log(f"Failed to register {name} into mmengine.{reg_name}: {_e}")
                debug_log(f"Synchronized {added} entries into mmengine.{reg_name} (inline)")
        except Exception as _e:
            debug_log(f"Registry sync failed in inline mode: {_e}")

        # Build and run
        try:
            runner = Runner.from_cfg(cfg)
            runner.test()
        except Exception:
            import traceback as _tb
            debug_log("Inline runner.test() raised an exception:\n" + _tb.format_exc())
            raise

        if not result_pkl_path.exists():
            raise FileNotFoundError(f"Result file not found: {result_pkl_path}")
        with open(result_pkl_path, "rb") as f:
            result_data = pickle.load(f)
        parsed_result = parse_test_result(result_data)
        debug_log(f"Inline test completed successfully: {parsed_result}")
        return parsed_result
    finally:
        # cleanup temporary files created during test
        try:
            if ann_pkl_path and Path(ann_pkl_path).exists():
                Path(ann_pkl_path).unlink()
                debug_log(f"Cleaned up: {ann_pkl_path}")
        except Exception as _e:
            debug_log(f"Failed to cleanup {ann_pkl_path}: {_e}")
        try:
            if result_pkl_path and Path(result_pkl_path).exists():
                Path(result_pkl_path).unlink()
                debug_log(f"Cleaned up: {result_pkl_path}")
        except Exception as _e:
            debug_log(f"Failed to cleanup {result_pkl_path}: {_e}")


def parse_test_result(result_data):
    """Parse result.pkl produced by DumpResults into a friendly dict.

    Returns a dict with keys: status, num_samples, predictions (list).
    Each prediction contains sample_index, scores (optional), predicted_class,
    and ground_truth_class (if present).
    """
    if not isinstance(result_data, list):
        debug_log(f"Unexpected result format: {type(result_data)}")
        return {
            "status": "error",
            "message": "Unexpected result format",
            "raw_type": str(type(result_data)),
        }

    if len(result_data) == 0:
        return {
            "status": "success",
            "num_samples": 0,
            "predictions": [],
        }

    # For this API we only need the model's prediction (binary mapping):
    # - treat predicted_class == 1 as True, predicted_class == 0 as False
    # - ignore ground-truth fields (client is using this for inference only)
    predictions = []
    for idx, item in enumerate(result_data):
        pred_bool = None
        # normalized predicted label extraction
        if 'pred_label' in item:
            try:
                pred_bool = bool(int(item['pred_label']) == 1)
            except Exception:
                pred_bool = None
        elif 'pred_labels' in item:
            try:
                pred_bool = bool(int(item['pred_labels']) == 1)
            except Exception:
                pred_bool = None
        elif 'pred_scores' in item:
            # fallback: if scores exist, pick argmax
            try:
                import numpy as _np
                scores = _np.asarray(item['pred_scores'])
                pred_idx = int(_np.argmax(scores))
                pred_bool = (pred_idx == 1)
            except Exception:
                pred_bool = None

        predictions.append({
            "sample_index": idx,
            "prediction": pred_bool,
        })

    result = {
        "status": "success",
        "num_samples": len(result_data),
        "predictions": predictions,
    }

    return result
