#!/usr/bin/env python3
import json
import os

cov_file = "/mnt/d/MERN/TinyTensor/artifacts/phase3/coverage/coverage.json"
if not os.path.exists(cov_file):
    print("Coverage file not found:", cov_file)
    exit(1)

with open(cov_file, "r") as f:
    data = json.load(f)

print("Root keys:", list(data.keys()))
if "files" in data:
    f0 = data["files"][0]
    print("File[0] keys:", list(f0.keys()))
    print("Sample file info:", {k: f0[k] for k in f0 if k not in ("lines", "branches", "functions")})
