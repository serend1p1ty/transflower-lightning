import sys
sys.path.append(".")

import joblib
import numpy as np
from analysis.pymo.preprocessing import *
from analysis.pymo.viz_tools import *
from analysis.pymo.writers import *


def feat2motion(
    feat=None,
    feat_path=None,
    scaler=None,
    scaler_path=None,
    pipeline=None,
    pipeline_path=None,
    video_path="render.mp4",
):
    """Restore the motion features to motion.

    Must specify [feat/scaler/pipeline] or [feat_path/scaler_path/pipeline_path]
    N: number of frames
    d: motion feature dimension

    Args:
        feat (np.array, optional): motion features, [N, d]. Defaults to None.
        feat_path (str, optional): path to motion features. Defaults to None.
        scaler (obj, optional): scaler object. Defaults to None.
        scaler_path (str, optional): path to scaler. Defaults to None.
        pipeline (obj, optional): pipeline obj. Defaults to None.
        pipeline_path (str, optional): path to pipeline obj. Defaults to None.
    """
    assert feat is not None or feat_path is not None
    assert scaler is not None or scaler_path is not None
    assert pipeline is not None or pipeline_path is not None

    if feat is None:
        feat = np.load(feat_path)
    if scaler is None:
        scaler = joblib.load(scaler_path)
    if pipeline is None:
        pipeline = joblib.load(pipeline_path)

    unscaled_feat = scaler.inverse_transform(feat)
    bvh_data = pipeline.inverse_transform([unscaled_feat])

    bvh2pos = MocapParameterizer("position")
    pos_data = bvh2pos.fit_transform(bvh_data)
    render_mp4(pos_data[0], video_path, axis_scale=300, elev=45, azim=45)
