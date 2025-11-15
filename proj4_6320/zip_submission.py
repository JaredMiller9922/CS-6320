# ---------------------------------------------------------------------
# CS 6320 - Project 4: Depth Estimation using Stereo
# Submission Packaging Script
# ---------------------------------------------------------------------
# This script creates a ZIP file containing all required files and
# directories listed in zip_dir_list.yml.
#
# Usage:
#   python3 zip_submission.py --uid u1234567
# ---------------------------------------------------------------------

import argparse
import os
import shutil
import yaml

def copy(src, dest, is_directory, required):
    """
    Copies files or directories safely into temp_submission folder.
    If a required file/directory is missing, aborts the submission.
    """
    ds = "Directory" if is_directory else "File"

    try:
        if is_directory:
            shutil.copytree(src, dest)
        else:
            shutil.copy(src, dest)
        print(f"{ds} copied: {src} → {dest}")
    except Exception as e:
        if required:
            print(f"[ERROR] {ds} '{src}' not copied. Reason: {e}")
            print("Exiting — submission not zipped.")
            shutil.rmtree("temp_submission", ignore_errors=True)
            exit(1)
        else:
            print(f"[WARNING] Optional {ds} '{src}' missing — skipping.")


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Parse UID argument
    # -----------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Create submission ZIP for Project 4")
    parser.add_argument("--uid", required=True, type=str, help="University ID (e.g., u1234567)")
    args = parser.parse_args()

    if not args.uid.startswith("u"):
        raise ValueError("UID must begin with 'u' (example: u1234567)")

    # -----------------------------------------------------------------
    # Load zip_dir_list.yml
    # -----------------------------------------------------------------
    with open("zip_dir_list.yml", "r") as f:
        dir_list = yaml.safe_load(f)

    # Optional keys fallback (avoid KeyErrors)
    optional_dirs = dir_list.get("optional_directories", [])
    optional_files = dir_list.get("optional_files", [])
    required_dirs = dir_list.get("required_directories", [])
    required_files = dir_list.get("required_files", [])

    # -----------------------------------------------------------------
    # Create temp_submission folder
    # -----------------------------------------------------------------
    shutil.rmtree("temp_submission", ignore_errors=True)
    os.mkdir("temp_submission")

    # Copy required directories
    for d in required_dirs:
        copy(d, os.path.join("temp_submission", d), is_directory=True, required=True)

    # Copy required files
    for f in required_files:
        copy(f, os.path.join("temp_submission", f), is_directory=False, required=True)

    # Copy optional directories
    for d in optional_dirs:
        copy(d, os.path.join("temp_submission", d), is_directory=True, required=False)

    # Copy optional files
    for f in optional_files:
        copy(f, os.path.join("temp_submission", f), is_directory=False, required=False)

    # -----------------------------------------------------------------
    # Create final ZIP archive
    # -----------------------------------------------------------------
    out_file = f"{args.uid}.zip"
    shutil.make_archive(args.uid, "zip", "temp_submission")

    # Validate size limit
    max_size_mb = 80
    zip_size = os.path.getsize(out_file) / (10**6)

    if zip_size > max_size_mb:
        os.remove(out_file)
        print(f"Submission NOT ZIPPED: size {zip_size:.2f} MB > {max_size_mb} MB limit")
    else:
        print(f"Submission successfully created: {out_file} ({zip_size:.2f} MB)")

    # Clean up temp folder
    shutil.rmtree("temp_submission", ignore_errors=True)