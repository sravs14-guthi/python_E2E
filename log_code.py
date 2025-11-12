from pathlib import Path
import logging

def setup_logging(module_name):
    # ✅ Always create a separate folder for logs
    LOG_DIR = Path(__file__).resolve().parent / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOG_DIR / f"{module_name}.log"

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger(module_name)
    return logger