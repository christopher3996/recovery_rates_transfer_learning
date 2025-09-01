# XGB Heterogeneous Model after Major Revision (08/2025)

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from xgboost import XGBRegressor
from models.base_model import BaseModel

def _is_numeric_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

class XGBoostModel(BaseModel):
    """
    XGBoost with transfer across heterogeneous feature sets.

    Pre-train on source; fine-tune on target with three cases:
      A) target schema == source schema           -> continue boosting (adds trees)
      B) target missing some source columns       -> fill missing from source stats, then continue boosting
      C) target has new (extra) columns           -> residual transfer (stage-2 fits residuals on target-native cols)
    """

    def __init__(self,
                 xgb_params: Optional[Dict] = None,
                 epochs: int = 50,
                 seed: int = 42,
                 deterministic: bool = True,
                 benchmark: bool = False,
                 **kwargs):
        super().__init__(seed=seed, deterministic=deterministic, benchmark=benchmark)

        # Support legacy MODEL_PARAMS key `params=...`
        legacy = kwargs.pop("params", None)
        xgb_params = dict(xgb_params or {})
        if legacy:
            xgb_params.update(legacy)

        self.epochs = int(epochs)
        self.xgb_params = xgb_params
        self.xgb_params.setdefault("n_estimators", self.epochs)
        self.xgb_params.setdefault("random_state", seed)

        # Stage-1 (source / aligned)
        self.model = XGBRegressor(enable_categorical=True, **self.xgb_params)

        # Stage-2 (residual booster on target-native schema when extra cols exist)
        self.stage2_model: Optional[XGBRegressor] = None
        self.stage2_params: Optional[Dict] = None

        # Source schema & fill values
        self.schema_cols: List[str] = []
        self.fill_values: Dict[str, object] = {}
        self.train_dtypes: Dict[str, str] = {}

        # Histories (for plots)
        self.train_loss_history: List[float] = []
        self.train_r2_history: List[float] = []
        self.finetune_loss_history: List[float] = []
        self.finetune_r2_history: List[float] = []

        # Optional: restrict residual model to new features only (otherwise uses all target cols)
        self.residual_features = "all"   # "all" | "new_only"

    # ---------------- helpers ----------------
    def _capture_schema(self, X: pd.DataFrame):
        self.schema_cols = list(X.columns)
        self.train_dtypes = {c: str(X[c].dtype) for c in self.schema_cols}
        self.fill_values = {}
        for c in self.schema_cols:
            s = X[c]
            if _is_numeric_dtype(s):
                self.fill_values[c] = float(np.nanmean(s.values)) if len(s) else 0.0
            else:
                if len(s):
                    try:
                        mv = s.mode(dropna=True).iloc[0]
                    except Exception:
                        mv = s.dropna().iloc[0] if s.dropna().size else ""
                else:
                    mv = ""
                self.fill_values[c] = mv

    def _align_to_source_schema(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # add missing
        for c in self.schema_cols:
            if c not in X.columns:
                X[c] = self.fill_values.get(c, 0.0)
        # drop extra and reorder
        return X[self.schema_cols]

    def _log_train_rmse(self, model, phase: str):
        """Grab RMSE history if the wrapper provides it; else leave empty."""
        try:
            er = model.evals_result()
            rmse = er.get("validation_0", {}).get("rmse", []) or er.get("train", {}).get("rmse", [])
        except Exception:
            rmse = []
        if phase == "pretrain":
            self.train_loss_history = rmse
        else:
            self.finetune_loss_history = rmse

    # ---------------- API ----------------
    def train(self, X_train: pd.DataFrame, y_train, X_eval=None, y_eval=None, **kwargs):
        # reset histories
        self.train_loss_history.clear(); self.train_r2_history.clear()
        self.finetune_loss_history.clear(); self.finetune_r2_history.clear()
        self.stage2_model = None
    
        # capture training schema for later alignment
        self._capture_schema(X_train)
    
        # --- NEW: if eval is given, align it to the training schema to avoid DMatrix mismatch ---
        eval_set = []
        if X_eval is not None and y_eval is not None:
            X_eval = self._align_to_source_schema(X_eval)   # <—— key line
            eval_set = [(X_train, y_train), (X_eval, y_eval)]
    
        # train fresh
        self.model.set_params(n_estimators=self.xgb_params.get("n_estimators", self.epochs))
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self._log_train_rmse(self.model, phase="pretrain")
    
        # optional one-liner to mirror your logs
        print("[XGB] Train Done.")


    def fine_tune(self, X_train: pd.DataFrame, y_train, X_eval=None, y_eval=None, **kwargs):
        add_ep = int(kwargs.get("additional_epochs", max(1, self.epochs // 2)))
    
        src = set(self.schema_cols)
        tgt = set(X_train.columns)
        extra = sorted(list(tgt - src))     # target-only new columns
        missing = sorted(list(src - tgt))   # source columns absent in target
    
        # --- status at start ---
        print(f"[XGB] finetune start: +{add_ep} trees, extras={len(extra)}, missing={len(missing)}")
    
        if len(extra) == 0:
            # ---- Cases A+B: continue boosting on aligned schema ----
            X_ft = self._align_to_source_schema(X_train)
            eval_set = []
            if X_eval is not None and y_eval is not None:
                X_eval_al = self._align_to_source_schema(X_eval)
                eval_set = [(X_ft, y_train), (X_eval_al, y_eval)]
    
            curr = int(self.model.get_params().get("n_estimators", self.epochs))
            self.model.set_params(n_estimators=curr + add_ep)
            booster = self.model.get_booster()
    
            self.model.fit(X_ft, y_train, eval_set=eval_set, verbose=False, xgb_model=booster)
            self._log_train_rmse(self.model, phase="finetune")
            self.stage2_model = None
    
            # --- status at end ---
            last = float(self.finetune_loss_history[-1]) if self.finetune_loss_history else float("nan")
            total = int(self.model.get_params().get("n_estimators", self.epochs))
            print(f"[XGB] finetune done ✓ continued boosting to {total} trees (last RMSE={last:.4f})")
            return
    
        # ---- Case C: residual transfer ----
        X1_tr = self._align_to_source_schema(X_train)
        y0_tr = self.model.predict(X1_tr)
        y_res_tr = np.asarray(y_train) - np.asarray(y0_tr)
    
        if X_eval is not None and y_eval is not None:
            X1_ev = self._align_to_source_schema(X_eval)
            y0_ev = self.model.predict(X1_ev)
            y_res_ev = np.asarray(y_eval) - np.asarray(y0_ev)
        else:
            X1_ev = y0_ev = y_res_ev = None
    
        use_cols = extra if self.residual_features == "new_only" else list(X_train.columns)
        X2_tr = X_train[use_cols].copy()
        X2_ev = X_eval[use_cols].copy() if X_eval is not None else None
    
        self.stage2_params = dict(self.xgb_params)
        self.stage2_params["n_estimators"] = add_ep
        self.stage2_model = XGBRegressor(enable_categorical=True, **self.stage2_params)
    
        eval_set = []
        if X2_ev is not None:
            eval_set = [(X2_tr, y_res_tr), (X2_ev, y_res_ev)]
    
        self.stage2_model.fit(X2_tr, y_res_tr, eval_set=eval_set, verbose=False)
        self._log_train_rmse(self.stage2_model, phase="finetune")
    
        # --- status at end ---
        last = float(self.finetune_loss_history[-1]) if self.finetune_loss_history else float("nan")
        used = "new_only" if self.residual_features == "new_only" else "all"
        print(f"[XGB] finetune done ✓ residual transfer with {len(extra)} new feat (using {used}), last RMSE={last:.4f}")


    def predict(self, X: pd.DataFrame):
        if self.stage2_model is not None:
            # residual prediction: y = y0 + y_res_hat
            X1 = self._align_to_source_schema(X)
            y0 = self.model.predict(X1)

            if self.residual_features == "new_only":
                use_cols = sorted(list(set(X.columns) - set(self.schema_cols)))
                X2 = X[use_cols].copy()
            else:
                X2 = X

            y_res = self.stage2_model.predict(X2)
            return y0 + y_res

        # no stage-2: ensure alignment (handles missing/drop extras)
        X1 = self._align_to_source_schema(X)
        return self.model.predict(X1)

    def save_model(self, filepath):
        joblib.dump(
            dict(
                model=self.model,
                stage2_model=self.stage2_model,
                schema_cols=self.schema_cols,
                fill_values=self.fill_values,
                train_dtypes=self.train_dtypes,
                params=self.xgb_params,
                stage2_params=self.stage2_params,
                residual_features=self.residual_features,
            ),
            filepath,
        )

    def load_model(self, filepath):
        if os.path.exists(filepath):
            blob = joblib.load(filepath)
            self.model = blob["model"]
            self.stage2_model = blob.get("stage2_model", None)
            self.schema_cols = blob.get("schema_cols", [])
            self.fill_values = blob.get("fill_values", {})
            self.train_dtypes = blob.get("train_dtypes", {})
            self.xgb_params = blob.get("params", self.xgb_params)
            self.stage2_params = blob.get("stage2_params", None)
            self.residual_features = blob.get("residual_features", "all")
        else:
            print(f"Model file {filepath} does not exist.")

    def get_epoch_metrics(self):
        out = {}
        if self.train_loss_history:
            out["loss_pretrain"] = self.train_loss_history
        if self.train_r2_history:
            out["r2_pretrain"] = self.train_r2_history
        if self.finetune_loss_history:
            out["loss_finetune"] = self.finetune_loss_history
        if self.finetune_r2_history:
            out["r2_finetune"] = self.finetune_r2_history
        return out
