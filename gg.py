import os
import re
from datetime import datetime

import pandas as pd
import win32com.client as win32
from win32com.client import constants


# ========= CONFIG: EDIT ONLY THIS =========

# Path to Excel file containing rules
CONFIG_FILE = r"R:\Config\email_move_rules.xlsx"
CONFIG_SHEET = "Rules"

# Optional: Outlook profile name. Usually leave as None.
OUTLOOK_PROFILE = None


# ========= HELPER FUNCTIONS =========

def sanitize_filename(text: str) -> str:
    """Make a safe filename/folder name."""
    if not text:
        text = "NoName"
    text = text.strip()
    text = re.sub(r'[\\/:\"*?<>|]+', "_", text)
    text = re.sub(r"\s+", " ", text)
    if len(text) > 120:
        text = text[:120].rstrip()
    return text or "NoName"


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def get_outlook_namespace():
    """Get Outlook MAPI namespace."""
    outlook = win32.Dispatch("Outlook.Application").GetNamespace("MAPI")
    if OUTLOOK_PROFILE:
        outlook.Logon(OUTLOOK_PROFILE)
    return outlook


def get_folder(root_folder, path_parts):
    folder = root_folder
    for part in path_parts:
        if not part:
            continue
        folder = folder.Folders[part]
    return folder


def get_mailbox_root(ns, mailbox_name: str):
    try:
        return ns.Folders[mailbox_name]
    except Exception:
        raise RuntimeError(
            f"Mailbox '{mailbox_name}' not found. Check the display name in Outlook."
        )


def save_mail_as_msg(mail_item, dir_path: str):
    ensure_dir(dir_path)

    dt = mail_item.SentOn or mail_item.ReceivedTime
    if isinstance(dt, datetime):
        ts = dt.strftime("%Y%m%d_%H%M%S")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    subject_part = sanitize_filename(mail_item.Subject or "No subject")
    filename = f"{ts}_{subject_part}.msg"

    full_path = os.path.join(dir_path, filename)
    mail_item.SaveAs(full_path, constants.olMSG)


def load_rules_from_excel():
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")

    df = pd.read_excel(CONFIG_FILE, sheet_name=CONFIG_SHEET)

    required_cols = [
        "Enabled",
        "RuleName",
        "SourceMailbox",
        "SourceFolderPath",
        "SenderEmail",
        "TargetMailbox",
        "TargetFolderPath",
        "SaveRoot",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in config: {missing}")

    rules = []

    for _, row in df.iterrows():
        enabled_raw = str(row.get("Enabled", "")).strip().upper()
        if enabled_raw not in ("Y", "YES", "TRUE", "1"):
            continue

        rule_name = str(row["RuleName"]).strip()
        source_mailbox = str(row["SourceMailbox"]).strip()
        source_folder_path_raw = str(row["SourceFolderPath"]).strip()
        sender_email = str(row["SenderEmail"]).strip()
        target_mailbox = str(row["TargetMailbox"]).strip()
        target_folder_path_raw = str(row["TargetFolderPath"]).strip()
        save_root = str(row["SaveRoot"]).strip()

        if not save_root:
            print(f"Skipping rule '{rule_name}' — SaveRoot is blank.")
            continue

        source_folder_parts = [
            p.strip() for p in source_folder_path_raw.split("\\") if p.strip()
        ]
        target_folder_parts = [
            p.strip() for p in target_folder_path_raw.split("\\") if p.strip()
        ]

        rules.append(
            {
                "rule_name": rule_name,
                "source_mailbox": source_mailbox,
                "source_folder_parts": source_folder_parts,
                "sender_email": sender_email,
                "target_mailbox": target_mailbox,
                "target_folder_parts": target_folder_parts,
                "save_root": save_root,
            }
        )

    return rules


def process_rule(ns, rule: dict) -> int:
    rule_name = rule["rule_name"]
    source_mailbox = rule["source_mailbox"]
    source_folder_parts = rule["source_folder_parts"]
    sender_email = rule["sender_email"]
    target_mailbox = rule["target_mailbox"]
    target_folder_parts = rule["target_folder_parts"]
    save_root = rule["save_root"]

    print(f"\n=== Rule: {rule_name} ===")
    print(f"  Source: {source_mailbox}\\{'\\'.join(source_folder_parts)}")
    print(f"  Filter: Sender = {sender_email}")
    print(f"  Target: {target_mailbox}\\{'\\'.join(target_folder_parts)}")
    print(f"  Save to: {save_root}")

    try:
        src_root = get_mailbox_root(ns, source_mailbox)
        src_folder = get_folder(src_root, source_folder_parts)

        tgt_root = get_mailbox_root(ns, target_mailbox)
        tgt_folder = get_folder(tgt_root, target_folder_parts)
    except Exception as e:
        print(f"  ✖ ERROR locating folders: {e}")
        return 0

    try:
        filtered_items = src_folder.Items.Restrict(
            f"[SenderEmailAddress] = '{sender_email}'"
        )
    except Exception as e:
        print(f"  ✖ ERROR applying Restrict filter: {e}")
        return 0

    count = filtered_items.Count
    print(f"  → Found {count} matching email(s)")

    moved_count = 0

    rule_save_dir = os.path.join(save_root, sanitize_filename(rule_name))
    ensure_dir(rule_save_dir)

    for i in range(count, 0, -1):
        item = filtered_items[i]
        if item.Class != constants.olMail:
            continue

        subject_display = sanitize_filename(item.Subject or "No subject")

        try:
            moved = item.Move(tgt_folder)
            moved.UnRead = True
            moved.Save()

            save_mail_as_msg(moved, rule_save_dir)

            print(f"   ✔ Moved: {subject_display}")
            moved_count += 1

        except Exception as e:
            print(f"   ✖ ERROR moving/saving '{subject_display}': {e}")

    print(f"  → Completed. Moved {moved_count} email(s).")
    return moved_count


# ========= MAIN =========

def main():
    print("Loading rules from Excel...")
    rules = load_rules_from_excel()

    if not rules:
        print("No enabled rules found or all rules missing SaveRoot.")
        return

    print(f"Loaded {len(rules)} enabled rule(s).\n")

    ns = get_outlook_namespace()

    total_moved = 0
    for rule in rules:
        moved = process_rule(ns, rule)
        total_moved += moved

    print(f"\nDone. Total moved emails across all rules: {total_moved}")


if __name__ == "__main__":
    main()
