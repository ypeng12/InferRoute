import sys
import os
import uvicorn

# Ensure project root is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from inferroute.main import app

if __name__ == "__main__":
    uvicorn.run("inferroute.main:app", host="0.0.0.0", port=7860, reload=False)
