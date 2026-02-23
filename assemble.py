#!/usr/bin/env python3
import os
import glob
import argparse

def assemble_file(folder_path, output_path, cleanup=True):
    """
    Reassemble all .part files found inside a given folder into a single output file.
    Automatically infers the base filename from the first .part file.
    By default, deletes the parts after successful reconstruction.

    Args:
        folder_path (str): Path to the folder containing .part files.
        output_path (str): Path to the folder where the final file will be saved.
        cleanup (bool): If True, deletes part files after successful assembly.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"❌ Folder not found: {folder_path}")

    if not os.path.isdir(output_path):
        print(f"📁 Output folder '{output_path}' not found — creating it.")
        os.makedirs(output_path, exist_ok=True)

    parts = sorted(glob.glob(os.path.join(folder_path, "*.part*")))
    if not parts:
        raise FileNotFoundError(f"❌ No .part files found in folder: {folder_path}")

    # Infer base filename (everything before .partX)
    first_part = os.path.basename(parts[0])
    base_name = first_part.split(".part")[0]
    final_path = os.path.join(output_path, base_name)

    print(f"🔍 Found {len(parts)} parts in '{folder_path}'")
    print(f"🧩 Assembling into '{final_path}' ...")

    # Merge all parts
    with open(final_path, 'wb') as out_file:
        for part in parts:
            with open(part, 'rb') as p:
                out_file.write(p.read())
            print(f"   ➕ Added {os.path.basename(part)}")

    print(f"✅ Reassembly complete! Saved as: {final_path}")

    # Optional cleanup
    if cleanup:
        print("🧹 Cleaning up part files...")
        for part in parts:
            try:
                os.remove(part)
                print(f"   🗑️ Deleted {os.path.basename(part)}")
            except Exception as e:
                print(f"   ⚠️ Could not delete {part}: {e}")
        print("✅ Cleanup complete!")

    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Reassemble .part files into a single .h5 file (cleanup ON by default)."
    )

    parser.add_argument(
        "--folder_path",
        type=str,
        help="Path to the folder containing .part files."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the folder where the merged .h5 file will be saved."
    )

    parser.add_argument(
        "--no-cleanup",
        action="store_false",
        dest="cleanup",
        help="Disable deleting .part files after assembly (cleanup is ON by default)."
    )

    args = parser.parse_args()

    assemble_file(
        folder_path=args.folder_path,
        output_path=args.output_path,
        cleanup=args.cleanup
    )


if __name__ == "__main__":
    main()

