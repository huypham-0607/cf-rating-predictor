"""
Codeforces API data collector

Purpose: fetch raw problem and contest data from the Codeforces API and save to disk.

Endpoints:
  - /api/problemset.problems  → problems list + per-problem solved counts
  - /api/contest.list         → contest metadata (name, type, start time, duration)
"""

import json
import time
from pathlib import Path
from datetime import datetime, timezone

import requests
import yaml

from src.utils import get_logger

logger = get_logger(__name__)

_CFG_PATH = Path("configs/collection.yaml")

# Loading config collection.yaml into a dictionary
def _load_cfg() -> dict:
    with open (_CFG_PATH) as f:
        return yaml.safe_load(f)

class CodeforcesAPICollector:
    BASE_URL = "https://codeforces.com/api"

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            cfg = _load_cfg()
        
        # Assigning values from the dictionary
        api_cfg = cfg["api"]
        self.rate_limit_delay: float = api_cfg["rate_limit_delay"]
        self.max_retries: int = api_cfg["max_retries"]
        self.backoff_base = api_cfg["backoff_base"]
        self.timeout = api_cfg["timeout"]
        self.raw_dir = Path(cfg["output"]["raw_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------
    # Public Interface
    # --------------------------------------------------------------------

    def fetch_problems(self, force: bool = False) -> dict:
        """Fetch all problems from /api/problemset.problems."""
        dest = self.raw_dir / "problems_api.json"
        if dest.exists() and not force:
            logger.info("problem_api.json already exists (pass force=True to re-fetch)")
            with open(dest) as f:
                return json.load(f)
        
        logger.info("Fetching problemset.problems ...")
        data = self._get("problemset.problems")
        self._save(data, dest)
        logger.info(
            "Saved %d problems and %d statistics records to %s",
            len(data.get("result", {}).get("problems", [])),
            len(data.get("result", {}).get("problemStatistics", [])),
            dest,
        )

        return data
    
    def fetch_contests(self, force: bool = False) -> dict: 
        """Fetch all non-gym contests from /api/contest.list."""
        dest = self.raw_dir / "contests_api.json"
        if dest.exists() and not force:
            logger.info("contests_api.json already exists (pass force=True to re-fetch)")
            with open(dest) as f:
                return json.load(f)
        
        logger.info("Fetching contest.list ...")
        data = self._get("contest.list", params={"gym": "false"})
        self._save(data, dest)
        logger.info("Saved %d contests to %s", len(data.get("result",[])), dest)
        return data

    # --------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        url =  f"{self.BASE_URL}/{endpoint}"
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") != "OK":
                    raise ValueError(f"API returned status = {data.get('status')} : {data.get('comment')}")
                time.sleep(self.rate_limit_delay)
                return data
            except (requests.RequestException, ValueError) as exc:
                wait = self.backoff_base ** attempt
                logger.warning("Attempt %d/%d failed: %s - retrying in %.1fs", attempt+1, self.max_retries, exc, wait)
        raise RuntimeError(f"All {self.max_retries} attempts failed for endpoint '{endpoint}'")
    
    @staticmethod
    def _save(data:dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"fetched_at": datetime.now(timezone.utc).isoformat()}
        with open(path, "w") as f:
            json.dump({"_meta": meta, **data}, f, indent=2)
