from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping

import pandas as pd
import numpy as np

from src.regional_agent.config import get_region_current_dir, get_region_data_root
from engine.model_predict import predict_outcomes

OUTPUT_DIR = Path("outputs")
