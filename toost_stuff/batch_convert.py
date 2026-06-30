import os
import sys
import json
import shutil
import argparse
import subprocess

def print_success(text): print(f"\033[92m[+] {text}\033[0m")
def print_info(text):    print(f"\033[94m[*] {text}\033[0m")
def print_warn(text):    print(f"\033[93m[-] {text}\033[0m")
def print_error(text):   print(f"\033[91m[!] {text}\033[0m")

EXE_NAME = "toost.exe" if sys.platform == "win32" else "toost"

def find_exe():
    candidates = [
        os.path.join("bin", EXE_NAME),
        os.path.join(".", EXE_NAME),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    found = shutil.which(EXE_NAME)
    if found:
        return found
    return None

def world_size(json_path):
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("object_count", 0) + data.get("ground_count", 0)
    except Exception:
        return 0

def load_tags_index(input_dir):
    # tags.json (from extract_mm2_bcd.py) maps each .bcd stem to its tag list.
    path = os.path.join(input_dir, "tags.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print_warn(f"Could not read tags index '{path}': {e}")
        return {}

def attach_tags(json_path, tags):
    # Toost doesn't know about tags (they aren't in the .bcd), so add them here.
    if not os.path.isfile(json_path):
        return
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        data["tags"] = tags
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print_warn(f"Could not attach tags to '{json_path}': {e}")

def batch_convert(exe, input_dir, output_dir, images_dir, min_objects,
                  remove_grid, objects_over_pipes):
    bcd_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".bcd")]
    if not bcd_files:
        print_info(f"No .bcd files found in: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    tags_index = load_tags_index(input_dir)
    print_info(f"Processing {len(bcd_files)} file(s) -> {output_dir} (json) / {images_dir} (images)")
    if tags_index:
        print_info(f"Loaded tags for {len(tags_index)} level(s) from tags.json")
    print("-" * 60)

    ok = skipped = failed = 0
    for filename in sorted(bcd_files):
        bcd_path  = os.path.join(input_dir, filename)
        stem      = os.path.splitext(filename)[0]
        ow_json   = os.path.join(output_dir, f"{stem}_overworld.json")
        sub_json  = os.path.join(output_dir, f"{stem}_subworld.json")
        ow_png    = os.path.join(images_dir, f"{stem}_overworld.png")
        sub_png   = os.path.join(images_dir, f"{stem}_subworld.png")

        cmd = [exe, "-p", bcd_path,
               "--overworldJson", ow_json, "--subworldJson", sub_json,
               "-o", ow_png, "-s", sub_png]

        if remove_grid:
            cmd.append("-r")
        if objects_over_pipes:
            cmd.append("-e")

        print(f"  {filename} ...", end=" ", flush=True)
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print("\033[91mFAILED\033[0m")
            if result.stderr:
                print(f"    {result.stderr.decode(errors='replace').strip()}")
            failed += 1
            continue

        # Tags apply to the whole level: overworld always, subworld if kept.
        tags = tags_index.get(stem)
        if tags is not None:
            attach_tags(ow_json, tags)

        # Remove subworld JSON/PNG if it's below the size threshold
        sub_size = world_size(sub_json)
        if sub_size < min_objects:
            os.remove(sub_json)
            if os.path.exists(sub_png):
                os.remove(sub_png)
            print(f"\033[92mOK\033[0m  \033[93m(subworld skipped: {sub_size} objects)\033[0m")
            skipped += 1
        else:
            if tags is not None:
                attach_tags(sub_json, tags)
            print("\033[92mOK\033[0m")

        ok += 1

    print("-" * 60)
    print_success(f"Done: {ok}/{len(bcd_files)} converted, {skipped} empty subworlds dropped, {failed} failed.")

if __name__ == "__main__":
    os.system("color")

    parser = argparse.ArgumentParser(description="Batch convert SMM2 .bcd level files to JSON and PNG images.")
    parser.add_argument("folder",               help="Folder containing .bcd files")
    parser.add_argument("-o", "--output",       help="JSON output folder (default: <folder>/json/)")
    parser.add_argument("--images-output",      help="PNG output folder (default: <folder>/images/)")
    parser.add_argument("--min-objects",        type=int, default=1,
                        help="Minimum object+ground count to keep a subworld (default: 1)")
    parser.add_argument("--remove-grid",        action="store_true", help="Render without grid")
    parser.add_argument("--objects-over-pipes", action="store_true", help="Render objects over pipes")
    args = parser.parse_args()

    exe = find_exe()
    if not exe:
        print_error(f"Could not find '{EXE_NAME}'.")
        print_info("Build it once by running these commands in an MSYS2 MinGW64 terminal:")
        print_info("  cd /c/Users/mckeonp/Documents/GitHub/toost")
        print_info("  mingw32-make BUILD=release")
        sys.exit(1)
    print_success(f"Using exe: {exe}")

    output_dir = args.output or os.path.join(args.folder, "json")
    images_dir = args.images_output or os.path.join(args.folder, "images")
    batch_convert(exe, args.folder, output_dir, images_dir,
                  min_objects=args.min_objects,
                  remove_grid=args.remove_grid,
                  objects_over_pipes=args.objects_over_pipes)
