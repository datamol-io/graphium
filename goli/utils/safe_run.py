from loguru import logger
import traceback as tb


class SafeRun:
    def __init__(self, name: str, raise_error: bool = True, verbose: int = 2) -> None:
        """
        Run some code with error handling and some printing, using the with statment.

        Example:
            In the example below, the `2+None`, an error will be caught and printed.
            ```
            with SafeRun(name="Addition that fails", raise_error=False):
                2 + None
            ```

        Parameters:
            name: Name of the code, used for printing
            raise_error: Whether to raise an error, or to catch it and print instead
            verbose: The level of verbosity
                0: Do not print anything
                1: Print only the traced stack when an error is caught and `raise_error` is False
                2: Print headers and footers at the start and exit of the with statement.
        """
        self.name = name
        self.raise_error = raise_error
        self.verbose = verbose

    def __enter__(self):
        """
        Print that the with-statement started, if `self.verbose >= 2`
        """
        if self.verbose >= 2:
            logger.info(f"\n------------ {self.name} STARTED ------------")

    def __exit__(self, type, value, traceback):
        """
        Handle the error. Raise it if `self.raise_error==True`, otherwise ignore it
        and print it if `self.verbose >= 1`. Also print that the with-statement is
        completed if `self.verbose >= 2`.
        """
        if traceback is not None:
            if self.raise_error:
                if self.verbose >= 1:
                    logger.error(f"------------ {self.name} ERROR: ------------")
                return False
            else:
                if self.verbose >= 1:
                    logger.error(f"------------ {self.name} ERROR: ------------\nERROR skipped. Traceback:\n")
                    logger.trace(print("".join(tb.format_exception(None, value, traceback))))
                return True
        else:
            if self.verbose >= 2:
                logger.info("\n------------ {self.name} COMPLETED ------------\n\n")
            return True
