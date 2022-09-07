from tabnanny import verbose
from loguru import logger
import traceback as tb

class SafeRun():
    def __init__(self, name: str, raise_error: bool = True, verbose: bool = True) -> None:
        self.name = name
        self.raise_error = raise_error
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            logger.info(f"\n------------ {self.name} STARTED ------------")


    def __exit__(self, type, value, traceback):
        if traceback is not None:
            if self.raise_error:
                logger.error(f"------------ {self.name} ERROR: ------------")
                return False
            else:
                if self.verbose:
                    logger.error(f"------------ {self.name} ERROR: ------------\nERROR skipped. Traceback:\n")
                    logger.trace(print(''.join(tb.format_exception(None, value, traceback))))
                return True
        else:
            if self.verbose:
                logger.info("\n------------ {self.name} COMPLETED ------------\n\n")
            return True
