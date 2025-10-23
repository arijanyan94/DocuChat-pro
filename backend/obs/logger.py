import json, time, os
from typing import Any, Dict

LOG_PATH = os.getenv("OBS_LOG_PATH", "runtime/requests.log")

def log_event(event: Dict[str, Any]) -> None:
	os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
	evt = {"ts": time.time(), **event}
	with open(LOG_PATH, 'a') as f:
		f.write(json.dumps(evt, ensure_ascii=False) + "\n")