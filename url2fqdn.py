#!/usr/bin/env python3

from urllib.parse import urlparse
from typing import Any, Union

def parseURL(url:str) -> dict[str,Any]:

    rtn = {}
    tmp = urlparse(url)
    for f in ['scheme', 'hostname', 'port', 'path', 'params', 'query', 'fragment', 'username', 'password' ]:
        v = getattr(tmp, f, None)
        rtn[f] = v
    return rtn


if __name__ == '__main__':

    import sys
    with open('/dev/stdin', encoding='utf-8') as fp:
         lines = fp.read().split()

    for l in lines:
      p = parseURL(l)
      print(p.get('hostname'))
