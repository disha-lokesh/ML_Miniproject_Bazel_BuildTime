"""Count the direct dependencies a source file declares."""
import re

JAVA_EXT = {".java"}
CPP_EXT = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}
CODE_EXT = JAVA_EXT | CPP_EXT

_JAVA = re.compile(rb"^\s*import\s+[\w.*]+\s*;", re.M)
_CPP = re.compile(rb"^\s*#\s*include\s*[<\"]", re.M)


def count_deps(content: bytes, ext: str) -> int:
    """Number of import statements (Java) or #include lines (C/C++); 0 for other files."""
    if ext in JAVA_EXT:
        return len(_JAVA.findall(content))
    if ext in CPP_EXT:
        return len(_CPP.findall(content))
    return 0
