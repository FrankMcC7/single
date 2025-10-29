#!/usr/bin/env python3
"""
Utility to reassemble files that were split by splitter.py.

Given a manifest JSON file produced by splitter.py, this script concatenates all
listed parts back into the original file and optionally verifies the SHA-256 hash.

If launched without arguments (for example, by double-clicking), a simple GUI flow
prompts for the manifest and destination directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

BUF_SIZE = 4 * 1024 * 1024  # 4 MB chunks


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for the given file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(BUF_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(manifest_path: Path) -> dict:
    try:
        data = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse manifest {manifest_path}: {exc}") from exc

    for required_key in ("original", "sha256", "parts"):
        if required_key not in data:
            raise ValueError(f"Manifest missing required key: {required_key}")
    if not isinstance(data["parts"], Iterable):
        raise ValueError("Manifest 'parts' entry must be a sequence of filenames.")
    return data


def assemble_from_manifest(
    manifest_path: Path,
    output_dir: Path | None = None,
    overwrite: bool = False,
    skip_verify: bool = False,
) -> Path:
    """
    Reassemble the file described by ``manifest_path``.

    Returns the path to the reconstructed file.
    """
    manifest_path = Path(manifest_path)
    manifest_data = _read_manifest(manifest_path)

    parts_directory = manifest_path.parent
    target_directory = Path(output_dir) if output_dir else parts_directory
    target_directory.mkdir(parents=True, exist_ok=True)

    target_path = target_directory / manifest_data["original"]
    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"Destination file {target_path} already exists. "
            "Use --overwrite to replace it."
        )

    part_names = list(manifest_data["parts"])
    if not part_names:
        raise ValueError("Manifest contains no parts to assemble.")

    with target_path.open("wb") as destination:
        for part_name in part_names:
            part_path = parts_directory / part_name
            if not part_path.is_file():
                raise FileNotFoundError(f"Missing part file: {part_path}")

            with part_path.open("rb") as part_handle:
                while True:
                    block = part_handle.read(BUF_SIZE)
                    if not block:
                        break
                    destination.write(block)

    if not skip_verify:
        reconstructed_hash = _sha256_file(target_path)
        expected_hash = manifest_data["sha256"]
        if reconstructed_hash != expected_hash:
            raise ValueError(
                "SHA-256 verification failed. "
                f"Expected {expected_hash}, got {reconstructed_hash}."
            )

    return target_path


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reassemble split files using a splitter.py manifest."
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        help="Path to the manifest JSON produced by splitter.py. "
        "Omit to launch the GUI picker.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional destination directory for the reassembled file. Defaults to the manifest directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip SHA-256 verification of the reconstructed file.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the GUI prompts (ignores other CLI options except --gui).",
    )
    return parser.parse_args(argv)


def _launch_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError as exc:  # pragma: no cover - python built without Tk
        raise RuntimeError(
            "Tkinter is not available in this Python installation; "
            "run the script with CLI arguments instead."
        ) from exc

    root = tk.Tk()
    root.withdraw()

    manifest_path = filedialog.askopenfilename(
        parent=root,
        title="Select manifest JSON",
        filetypes=[("Manifest JSON", "*.json"), ("All files", "*.*")],
    )
    if not manifest_path:
        root.destroy()
        return

    output_dir = filedialog.askdirectory(
        parent=root,
        title="Select output folder (Cancel to use manifest folder)",
    )
    output_dir_path = Path(output_dir) if output_dir else None

    def _show_error(message: str) -> None:
        messagebox.showerror("Reassembly failed", message, parent=root)

    try:
        target_path = assemble_from_manifest(
            manifest_path=Path(manifest_path),
            output_dir=output_dir_path,
            overwrite=False,
            skip_verify=False,
        )
    except FileExistsError:
        overwrite = messagebox.askyesno(
            parent=root,
            title="File exists",
            message=(
                "The destination file already exists.\n"
                "Do you want to overwrite it?"
            ),
        )
        if not overwrite:
            root.destroy()
            return
        try:
            target_path = assemble_from_manifest(
                manifest_path=Path(manifest_path),
                output_dir=output_dir_path,
                overwrite=True,
                skip_verify=False,
            )
        except Exception as exc:  # pylint: disable=broad-except
            _show_error(str(exc))
            root.destroy()
            return
    except Exception as exc:  # pylint: disable=broad-except
        _show_error(str(exc))
        root.destroy()
        return

    messagebox.showinfo(
        parent=root,
        title="Success",
        message=f"Reassembled file written to:\n{target_path}\nHash verified.",
    )
    root.destroy()


def main(argv: list[str] | None = None) -> None:
    args = _parse_arguments(argv)

    if args.gui or args.manifest is None:
        try:
            _launch_gui()
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        return

    target_path = assemble_from_manifest(
        manifest_path=args.manifest,
        output_dir=args.out_dir,
        overwrite=args.overwrite,
        skip_verify=args.skip_verify,
    )
    print(f"Reassembled file written to: {target_path}")
    if not args.skip_verify:
        print("SHA-256 verification succeeded.")


if __name__ == "__main__":
    main()
