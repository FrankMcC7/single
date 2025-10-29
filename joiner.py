#!/usr/bin/env pythonw
"""
Simple GUI wrapper around join_split_parts.py for one-click usage.
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from join_split_parts import assemble_from_manifest


def _select_manifest(root: tk.Tk) -> Path | None:
    path = filedialog.askopenfilename(
        parent=root,
        title="Select manifest JSON",
        filetypes=[("Manifest JSON", "*.json"), ("All files", "*.*")],
    )
    return Path(path) if path else None


def _select_output_dir(root: tk.Tk) -> Path | None:
    path = filedialog.askdirectory(
        parent=root,
        title="Select output folder (Cancel to use manifest folder)",
    )
    return Path(path) if path else None


def main() -> None:
    root = tk.Tk()
    root.withdraw()

    manifest_path = _select_manifest(root)
    if manifest_path is None:
        return

    output_dir = _select_output_dir(root)

    try:
        target_path = assemble_from_manifest(
            manifest_path=manifest_path,
            output_dir=output_dir,
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
            return
        try:
            target_path = assemble_from_manifest(
                manifest_path=manifest_path,
                output_dir=output_dir,
                overwrite=True,
                skip_verify=False,
            )
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Reassembly failed", str(exc))
            return
    except Exception as exc:  # pylint: disable=broad-except
        messagebox.showerror("Reassembly failed", str(exc))
        return

    messagebox.showinfo(
        parent=root,
        title="Success",
        message=f"Reassembled file written to:\n{target_path}\nHash verified.",
    )


if __name__ == "__main__":
    main()
