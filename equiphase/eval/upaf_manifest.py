import os
import sys
import json
import time
import hashlib
import platform
import importlib
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime, timezone

def _ver(name):
    try:
        return importlib.import_module(name).__version__
    except Exception:
        return "NOT_INSTALLED"

class IntegrityViolation(Exception):
    """Raised when TAMPER, LEDGER_RECORD_TAMPERED, or LEDGER_CHAIN_BROKEN verdict is triggered."""
    def __init__(self, verdict: str, changed_keys: list):
        self.verdict = verdict
        self.changed_keys = changed_keys
        super().__init__(f"Integrity Violation ({verdict}): Sealed keys altered -> {changed_keys}")

class UPAFManifest:
    """
    UPAF Hash Manifest v1 Implementation
    Strict adherence to UPAF Hash Manifest Specification v1 & Rule 3 Enforcement.
    Includes Cryptographic Hash Chaining (prev_manifest_self_sha256).
    """
    def __init__(self, task_id: str, task_type: str, entry_script: str):
        if task_type not in ["binary", "multiclass", "regression"]:
            raise ValueError(f"Task type must be explicitly specified as 'binary', 'multiclass', or 'regression'. Received: '{task_type}'")
            
        self.manifest_version = "1.0"
        self.task_id = task_id
        self.task_type = task_type
        self.entry_script = entry_script
        self.created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.run_id = f"upaf_{task_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{hashlib.sha256(self.created_utc.encode()).hexdigest()[:4]}"
        
        self.sealed = {}
        self.outputs = {}
        self.approved_change_reason = self._load_approval_file()

    def _load_approval_file(self) -> str:
        """Load approval reason exclusively from external approvals JSON file (Rule 3 Enforcement)."""
        approvals_dir = r"C:\Project\AI\EquiPhase\approvals"
        approval_file = os.path.join(approvals_dir, f"{self.task_id}.json")
        if os.path.exists(approval_file):
            try:
                with open(approval_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("approved_change_reason", None)
            except Exception:
                return None
        return None

    @staticmethod
    def canon_array(a: np.ndarray) -> bytes:
        """Canonical Serialization for NumPy arrays."""
        a = np.ascontiguousarray(a)
        
        if a.dtype.kind in ('O', 'U', 'S'):
            payload = json.dumps([None if x is None else str(x) for x in a.ravel()],
                                 ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            header = json.dumps({"dtype": "str", "shape": list(a.shape)},
                                sort_keys=True, separators=(',', ':')).encode('utf-8')
            return b"NPSTR\x00" + header + b"\x00" + payload
            
        if a.dtype.kind == 'f':
            a = a.copy()
            a[np.isnan(a)] = np.float64(np.nan).astype(a.dtype)
            a[a == 0.0] = 0.0
            
        header = json.dumps({
            "dtype": a.dtype.str,
            "shape": list(a.shape),
            "order": "C",
        }, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return b"NPARR\x00" + header + b"\x00" + a.tobytes()

    @staticmethod
    def canon_frame(df: pd.DataFrame) -> bytes:
        """Canonical Serialization for DataFrames."""
        meta = json.dumps({
            "columns": list(df.columns),
            "dtypes": [str(t) for t in df.dtypes],
        }, sort_keys=True, separators=(',', ':')).encode('utf-8')
        body = b"".join(UPAFManifest.canon_array(df[c].to_numpy()) for c in df.columns)
        return b"DFRAME\x00" + meta + b"\x00" + body

    @staticmethod
    def canon_json(obj) -> bytes:
        """Canonical Serialization for Dict / JSON."""
        return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')

    @staticmethod
    def canon_file(path: str) -> bytes:
        """Canonical Serialization for files (CRLF -> LF normalized)."""
        with open(path, 'rb') as f:
            content = f.read()
        return content.replace(b'\r\n', b'\n')

    def seal_data(self, X: np.ndarray, y: np.ndarray, confounds: np.ndarray = None, 
                  feature_names: list = None, provenance: dict = None, groups: np.ndarray = None, raw_files: list = None):
        """
        Seal Data Layer according to §2.1.
        Rule 3 Enforcement: raw_files MUST be a non-empty list of existing physical dataset files.
        """
        if not raw_files or not isinstance(raw_files, list) or len(raw_files) == 0:
            raise ValueError("[RULE 3 INTEGRITY VIOLATION] seal_data requires a non-empty raw_files list pointing to existing physical dataset files on disk!")
            
        rf_list = []
        for rf in raw_files:
            if not os.path.exists(rf):
                raise ValueError(f"[RULE 3 INTEGRITY VIOLATION] raw_file path does not exist on disk: '{rf}'")
            rf_bytes = self.canon_file(rf)
            rf_list.append({
                "path": rf,
                "bytes": len(rf_bytes),
                "sha256": hashlib.sha256(rf_bytes).hexdigest()
            })
        self.sealed["raw_files"] = rf_list

        if self.task_type == 'binary':
            unique_y = set(np.unique(y))
            if not unique_y.issubset({0, 1, 0.0, 1.0}):
                raise ValueError(f"Task type is 'binary' but y contains values outside {{0, 1}}: {unique_y}")

        self.sealed["X"] = {
            "sha256": hashlib.sha256(self.canon_array(X)).hexdigest(),
            "shape": list(X.shape),
            "dtype": str(X.dtype)
        }
        self.sealed["y"] = {
            "sha256": hashlib.sha256(self.canon_array(y)).hexdigest(),
            "shape": list(y.shape),
            "dtype": str(y.dtype)
        }
        
        if confounds is not None:
            if confounds.ndim == 1 and np.array_equal(confounds, y):
                raise ValueError("[RULE INTEGRITY VIOLATION] Label array y cannot be placed inside confounds matrix!")
            if confounds.ndim > 1:
                for col_i in range(confounds.shape[1]):
                    if np.array_equal(confounds[:, col_i], y):
                        raise ValueError(f"[RULE INTEGRITY VIOLATION] Label array y detected in confounds column {col_i}!")
                        
            self.sealed["confounds"] = {
                "sha256": hashlib.sha256(self.canon_array(confounds)).hexdigest(),
                "shape": list(confounds.shape)
            }
        else:
            self.sealed["confounds"] = None

        if feature_names is not None:
            self.sealed["feature_names"] = {
                "sha256": hashlib.sha256(self.canon_json(feature_names)).hexdigest()
            }

        if provenance is not None:
            self.sealed["provenance"] = {
                "sha256": hashlib.sha256(self.canon_json(provenance)).hexdigest()
            }

        if groups is not None:
            self.sealed["groups"] = {
                "sha256": hashlib.sha256(self.canon_array(groups)).hexdigest()
            }

    def seal_splits(self, split_indices: dict, split_rule: str):
        """Seal Split Layer according to §2.2 with int64 pinning."""
        serialized_splits = {k: self.canon_array(np.asarray(v, dtype=np.int64)) for k, v in split_indices.items()}
        combined_bytes = b"".join(serialized_splits.values())
        self.sealed["split_indices"] = {
            "sha256": hashlib.sha256(combined_bytes).hexdigest(),
            "n_folds": len(split_indices)
        }
        self.sealed["split_rule"] = {
            "sha256": hashlib.sha256(split_rule.encode('utf-8')).hexdigest(),
            "rule": split_rule
        }

    def seal_code_env(self, local_module_paths: list = None, repo_cwd: str = None):
        """Seal Code & Environment Layer with dynamic git dirty status."""
        if os.path.exists(self.entry_script):
            script_bytes = self.canon_file(self.entry_script)
            self.sealed["entry_script"] = {
                "path": self.entry_script,
                "sha256": hashlib.sha256(script_bytes).hexdigest()
            }
        else:
            self.sealed["entry_script"] = {"path": self.entry_script, "sha256": "FILE_NOT_FOUND"}

        lm_list = []
        if local_module_paths:
            for lmp in local_module_paths:
                if os.path.exists(lmp):
                    lm_bytes = self.canon_file(lmp)
                    lm_list.append({
                        "path": lmp,
                        "sha256": hashlib.sha256(lm_bytes).hexdigest()
                    })
        self.sealed["local_modules"] = lm_list

        self.sealed["runtime"] = {
            "python": sys.version.split()[0],
            "numpy": _ver("numpy"),
            "torch": _ver("torch"),
            "sklearn": _ver("sklearn"),
            "scipy": _ver("scipy"),
            "pandas": _ver("pandas"),
            "platform": platform.platform()
        }

        repo_dir = repo_cwd if repo_cwd and os.path.exists(repo_cwd) else r"C:\Project\AI\EquiPhase"
        try:
            h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True)
            st = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True)
            git_hash = h.stdout.strip() if h.returncode == 0 else "NOT_GIT_REPO"
            dirty = bool(st.stdout.strip()) if st.returncode == 0 else None
            if dirty:
                print(f"[WARNING] Git working directory is DIRTY ({repo_dir}). Uncommitted changes exist!")
        except Exception:
            git_hash, dirty = "NOT_GIT_REPO", None

        self.sealed["git_commit"] = {"hash": git_hash, "dirty": dirty}

    def seal_execution(self, model_config: dict, seeds: list, metric_definition: dict):
        """Seal Execution Layer according to §2.4."""
        self.sealed["model_config"] = {
            "sha256": hashlib.sha256(self.canon_json(model_config)).hexdigest(),
            "class": model_config.get("class", "Unknown")
        }
        self.sealed["seeds"] = seeds
        self.sealed["metric_definition"] = metric_definition

    def seal_outputs(self, predictions_path: str = None, results_path: str = None):
        """Seal Output Layer into self.sealed['outputs'] according to §2.5."""
        if predictions_path and os.path.exists(predictions_path):
            pred_bytes = self.canon_file(predictions_path)
            self.outputs["predictions_file"] = {
                "path": predictions_path,
                "sha256": hashlib.sha256(pred_bytes).hexdigest()
            }
            
        if results_path and os.path.exists(results_path):
            res_bytes = self.canon_file(results_path)
            self.outputs["results_file"] = {
                "path": results_path,
                "sha256": hashlib.sha256(res_bytes).hexdigest()
            }
            
        self.sealed["outputs"] = self.outputs

    def build_manifest_json(self, open_count: int = 1, status: str = "FIRST_SEAL", history: list = None, prev_hash: str = "0"*64) -> dict:
        """Build manifest object with self hash and cryptographic hash chaining."""
        manifest_history = history if history is not None else [
            {"utc": self.created_utc, "status": status, "changed_keys": []}
        ]
        
        manifest_dict = {
            "manifest_version": "1.0",
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "sealed": self.sealed,
            "outputs": self.outputs,
            "seal_status": status,
            "open_count": open_count,
            "history": manifest_history,
            "approved_change_reason": self.approved_change_reason,
            "prev_manifest_self_sha256": prev_hash
        }
        
        sealed_json_bytes = self.canon_json(manifest_dict)
        manifest_dict["manifest_self_sha256"] = hashlib.sha256(sealed_json_bytes).hexdigest()
        return manifest_dict


class UPAFEvaluator:
    """
    UPAF Evaluator with Non-Mutating verify_only Inspector, Cryptographic Hash Chaining, and Repository Ledger Anchoring.
    """
    REPO_LEDGER_PATH = r"C:\Project\AI\EquiPhase\ledger\operational_manifest_ledger.jsonl"
    REPO_TIP_PATH = r"C:\Project\AI\EquiPhase\ledger\ledger_tip.sha256"

    @staticmethod
    def save_prediction_persistence(predictions_df: pd.DataFrame, output_path: str):
        """Save raw predictions according to §5 specification."""
        required_cols = ["sample_id", "group_id", "fold", "seed", "split", "y_true", "y_score", "y_pred", "protocol"]
        for col in required_cols:
            if col not in predictions_df.columns:
                raise ValueError(f"Predictions DataFrame missing required column: '{col}'")
                
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions_df.to_csv(output_path, index=False)

    @staticmethod
    def save_tip_anchor(ledger_path: str = None, tip_file_path: str = None):
        """Save the tip manifest_self_sha256 of the ledger log to an external anchor file."""
        lp = ledger_path if ledger_path else UPAFEvaluator.REPO_LEDGER_PATH
        tp = tip_file_path if tip_file_path else UPAFEvaluator.REPO_TIP_PATH
        if os.path.exists(lp):
            with open(lp, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                last_rec = json.loads(lines[-1])
                tip_sha = last_rec.get("manifest_self_sha256", "0"*64)
                os.makedirs(os.path.dirname(tp), exist_ok=True)
                with open(tp, "w", encoding="utf-8") as tf:
                    tf.write(tip_sha + "\n")

    @staticmethod
    def append_ledger(manifest_dict: dict, ledger_path: str):
        """Append manifest to ledger log in STRICT APPEND MODE ONLY ('a')."""
        os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(manifest_dict, sort_keys=True, ensure_ascii=False) + "\n")

    @staticmethod
    def verify_only(ledger_path: str = None, tip_file_path: str = None) -> dict:
        """
        PURE NON-MUTATING INSPECTOR.
        Inspects ledger line-by-line without EVER appending to disk.
        """
        lp = ledger_path if ledger_path else UPAFEvaluator.REPO_LEDGER_PATH
        tp = tip_file_path if tip_file_path else UPAFEvaluator.REPO_TIP_PATH
        
        if not os.path.exists(lp):
            return {"valid": False, "reason": "LEDGER_NOT_FOUND", "line_reports": []}
            
        line_reports = []
        last_sha256 = "0" * 64
        all_valid = True
        
        with open(lp, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            
        has_genesis = any(json.loads(l).get("seal_status") == "GENESIS" for l in lines)
        genesis_reached = not has_genesis  # If no GENESIS record exists, entire ledger is post-GENESIS
        
        for line_idx, line in enumerate(lines, 1):
            rec = json.loads(line)
            status = rec.get("seal_status")
            task_id = rec.get("task_id")
            
            if status == "GENESIS":
                genesis_reached = True
                line_reports.append({"line": line_idx, "task_id": task_id, "status": status, "verification": "GENESIS_CHECKPOINT"})
                last_sha256 = rec.get("manifest_self_sha256", last_sha256)
                continue
                
            if not genesis_reached:
                line_reports.append({"line": line_idx, "task_id": task_id, "status": status, "verification": "unverifiable_legacy"})
                last_sha256 = rec.get("manifest_self_sha256", last_sha256)
                continue
                
            # Post-GENESIS Strict Chain Link Verification
            if "prev_manifest_self_sha256" in rec:
                stored_prev = rec.get("prev_manifest_self_sha256")
                if stored_prev != last_sha256 and last_sha256 != "0"*64:
                    line_reports.append({"line": line_idx, "task_id": task_id, "status": status, "verification": "LEDGER_CHAIN_BROKEN"})
                    all_valid = False
                    continue
                    
            # Post-GENESIS Strict Self-Hash Verification
            if "manifest_self_sha256" in rec:
                stored_hash = rec["manifest_self_sha256"]
                rec_copy = rec.copy()
                del rec_copy["manifest_self_sha256"]
                h_canon = hashlib.sha256(UPAFManifest.canon_json(rec_copy)).hexdigest()
                h_std = hashlib.sha256(json.dumps(rec_copy, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
                
                if stored_hash not in (h_canon, h_std):
                    line_reports.append({"line": line_idx, "task_id": task_id, "status": status, "verification": "LEDGER_RECORD_TAMPERED"})
                    all_valid = False
                    continue
                    
            line_reports.append({"line": line_idx, "task_id": task_id, "status": status, "verification": "VALID"})
            last_sha256 = rec.get("manifest_self_sha256", last_sha256)
                    
        # Tip anchor check
        if tp and os.path.exists(tp):
            with open(tp, "r", encoding="utf-8") as tf:
                anchored_tip = tf.read().strip()
            if anchored_tip != last_sha256:
                all_valid = False
                line_reports.append({"line": "TIP_ANCHOR", "verification": "EXTERNAL_TIP_MISMATCH", "anchored": anchored_tip, "actual": last_sha256})

        return {"valid": all_valid, "line_reports": line_reports, "tip_sha256": last_sha256}

    @staticmethod
    def verify_and_update_lock(manifest_obj: UPAFManifest, ledger_path: str = None) -> dict:
        """Verify lock, verify self-hashes, apply cryptographic hash chaining, handle run_id invalidations."""
        lp = ledger_path if ledger_path else UPAFEvaluator.REPO_LEDGER_PATH
        
        if not os.path.exists(lp):
            final_json = manifest_obj.build_manifest_json(open_count=1, status="FIRST_SEAL", prev_hash="0"*64)
            UPAFEvaluator.append_ledger(final_json, lp)
            return {"verdict": "FIRST_SEAL", "violation": False, "changed_keys": []}
            
        previous_records_post_genesis = []
        invalidated_run_ids = set()
        last_sha256 = "0" * 64
        
        with open(lp, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            
        has_genesis = any(json.loads(l).get("seal_status") == "GENESIS" for l in lines)
        genesis_reached = not has_genesis  # If no GENESIS record exists, entire ledger is post-GENESIS
        
        for line_idx, line in enumerate(lines, 1):
            rec = json.loads(line)
            status = rec.get("seal_status")
            
            if status == "GENESIS":
                genesis_reached = True
                last_sha256 = rec.get("manifest_self_sha256", last_sha256)
                continue
                
            if status == "INVALIDATED":
                inv_run_id = rec.get("invalidates_run_id")
                if inv_run_id:
                    invalidated_run_ids.add(inv_run_id)
                    
            if not genesis_reached:
                last_sha256 = rec.get("manifest_self_sha256", last_sha256)
                continue
                
            # Post-GENESIS Cryptographic Hash Chaining Link Check
            if "prev_manifest_self_sha256" in rec:
                stored_prev = rec.get("prev_manifest_self_sha256")
                if stored_prev != last_sha256 and last_sha256 != "0"*64:
                    return {"verdict": "LEDGER_CHAIN_BROKEN", "violation": True, "changed_keys": [f"ledger_line_{line_idx}"]}
                    
            # Post-GENESIS Strict Self-Hash Verification (No hash_a fallback!)
            if "manifest_self_sha256" in rec:
                stored_hash = rec["manifest_self_sha256"]
                rec_copy = rec.copy()
                del rec_copy["manifest_self_sha256"]
                h_canon = hashlib.sha256(UPAFManifest.canon_json(rec_copy)).hexdigest()
                h_std = hashlib.sha256(json.dumps(rec_copy, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
                
                if stored_hash not in (h_canon, h_std):
                    return {"verdict": "LEDGER_RECORD_TAMPERED", "violation": True, "changed_keys": [f"ledger_line_{line_idx}"]}
                        
            previous_records_post_genesis.append(rec)
            last_sha256 = rec.get("manifest_self_sha256", last_sha256)
                    
        # Filter matching previous records ONLY from post-GENESIS valid records
        matching_prev = [
            r for r in previous_records_post_genesis 
            if r.get("task_id") == manifest_obj.task_id 
            and r.get("seal_status") not in ("TAMPER_REJECTED", "INVALIDATED", "GENESIS", "LEDGER_OVERWRITE_INCIDENT")
            and r.get("run_id") not in invalidated_run_ids
        ]
        
        if not matching_prev:
            final_json = manifest_obj.build_manifest_json(open_count=1, status="FIRST_SEAL", prev_hash=last_sha256)
            UPAFEvaluator.append_ledger(final_json, lp)
            return {"verdict": "FIRST_SEAL", "violation": False, "changed_keys": []}
            
        latest_prev = matching_prev[-1]
        
        prev_sealed = latest_prev.get("sealed", {})
        new_sealed = manifest_obj.sealed
        
        changed_keys = []
        for k in set(prev_sealed.keys()).union(set(new_sealed.keys())):
            if prev_sealed.get(k) != new_sealed.get(k):
                changed_keys.append(k)
                
        prev_count = latest_prev.get("open_count", 1)
        prev_hist = latest_prev.get("history", [])
        
        if not changed_keys:
            new_count = prev_count + 1
            new_hist = prev_hist + [{"utc": manifest_obj.created_utc, "status": "REOPEN", "changed_keys": []}]
            final_json = manifest_obj.build_manifest_json(open_count=new_count, status="REOPEN", history=new_hist, prev_hash=last_sha256)
            UPAFEvaluator.append_ledger(final_json, lp)
            return {"verdict": "REOPEN", "violation": False, "changed_keys": []}
            
        if set(changed_keys) <= {"runtime"}:
            new_count = prev_count + 1
            new_hist = prev_hist + [{"utc": manifest_obj.created_utc, "status": "BENIGN_DRIFT", "changed_keys": changed_keys}]
            final_json = manifest_obj.build_manifest_json(open_count=new_count, status="BENIGN_DRIFT", history=new_hist, prev_hash=last_sha256)
            UPAFEvaluator.append_ledger(final_json, lp)
            return {"verdict": "BENIGN_DRIFT", "violation": False, "changed_keys": changed_keys}
            
        if manifest_obj.approved_change_reason:
            new_count = prev_count + 1
            new_hist = prev_hist + [{"utc": manifest_obj.created_utc, "status": "SCOPE_CHANGE", "changed_keys": changed_keys}]
            final_json = manifest_obj.build_manifest_json(open_count=new_count, status="SCOPE_CHANGE", history=new_hist, prev_hash=last_sha256)
            UPAFEvaluator.append_ledger(final_json, lp)
            return {"verdict": "SCOPE_CHANGE", "violation": False, "changed_keys": changed_keys, "reason": manifest_obj.approved_change_reason}
            
        rejected_json = manifest_obj.build_manifest_json(open_count=prev_count, status="TAMPER_REJECTED", prev_hash=last_sha256)
        UPAFEvaluator.append_ledger(rejected_json, lp)
        return {"verdict": "TAMPER", "violation": True, "changed_keys": changed_keys}
