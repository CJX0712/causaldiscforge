"""Structured error codes (E100-E500)."""


class ForgeError(Exception):
    code = "E000"
    def __init__(self, msg: str = ""):
        self.msg = msg
        super().__init__(f"[{self.code}] {msg}")


class E100DataError(ForgeError):
    code = "E100"  # invalid input data (NaN, non-finite, wrong shape)


class E200GraphError(ForgeError):
    code = "E200"  # invalid graph (not a valid DAG/CPDAG, cycle, bad codes)


class E300ConfigError(ForgeError):
    code = "E300"  # invalid configuration / parameter


class E400BackendError(ForgeError):
    code = "E400"  # SOTA backend unavailable or failed


class E500ConsensusError(ForgeError):
    code = "E500"  # consensus assembly failure
