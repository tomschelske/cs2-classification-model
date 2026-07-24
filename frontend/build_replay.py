"""Inject round.json into the replay HTML template -> frontend/replay.html."""
import json, sys
from pathlib import Path

data = Path(sys.argv[1]).read_text()  # round.json (compact)
tpl = Path(sys.argv[2]).read_text()   # template.html
out = Path(sys.argv[3])
html = '<meta charset="utf-8">\n' + tpl.replace("/*__DATA__*/", data)
out.write_text(html, encoding="utf-8")
print("wrote", out, f"({out.stat().st_size} bytes)")
