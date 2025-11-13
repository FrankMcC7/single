import os
import re
import time
import logging
from datetime import datetime

import pandas as pd
import win32com.client as win32
from win32com.client import constants


# ========= CONFIG: EDIT THESE =========

# Path to Excel file containing rules
CONFIG_FILE = r"R:\Config\email_move_rules.xlsx"
CONFIG_SHEET = "Rules"

# Directory where a log file will be created for each run
LOG_DIR = r"R:\Logs\EmailMover"

# Optional: Outlook profile name. Usually leave as None.
OUTLOOK_PROFILE = None

# Outlook MailItem class numeric value (MailItem = 43)
OL_MAILITEM_CLASS = 43


# ========= LOGGING SETUP =========

def setup_logging() -> str:
    """
    Configure logging to write both to console and to a per-run log file.
    Returns the log file path.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"email_move_{run_ts}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Log file: {log_path}")
    return log_path


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
    """Walk down a folder path like ['Inbox', 'Subfolder']."""
    folder = root_folder
    for part in path_parts:
        if not part:
            continue
        folder = folder.Folders[part]
    return folder


def get_mailbox_root(ns, mailbox_name: str):
    """Return the root folder of a mailbox by display name."""
    try:
        return ns.Folders[mailbox_name]
    except Exception as e:
        raise RuntimeError(
            f"Mailbox '{mailbox_name}' not found. Check display name in Outlook. "
            f"Underlying error: {e}"
        )


# ---- Period parsing helpers ----

MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def get_period_folder(base_dir: str, subject: str) -> str:
    """
    From subject text, detect 'Month YYYY' (e.g. 'October 2025') and return
    a folder path: base_dir\YYYY\MM-MonthName

    If pattern not found, returns base_dir\_UnknownPeriod
    """
    if subject is None:
        subject = ""

    pattern = (
        r"\b("
        r"January|February|March|April|May|June|July|August|September|October|November|December"
        r")\s+(\d{4})\b"
    )
    m = re.search(pattern, subject, flags=re.IGNORECASE)

    if not m:
        unknown_dir = os.path.join(base_dir, "_UnknownPeriod")
        ensure_dir(unknown_dir)
        return unknown_dir

    month_name_raw = m.group(1)
    year = m.group(2)

    month_key = month_name_raw.lower()
    month_num = MONTH_MAP.get(month_key)

    if not month_num:
        unknown_dir = os.path.join(base_dir, "_UnknownPeriod")
        ensure_dir(unknown_dir)
        return unknown_dir

    month_name = month_key.capitalize()
    year_dir = os.path.join(base_dir, year)
    month_dir_name = f"{month_num:02d}-{month_name}"
    month_dir = os.path.join(year_dir, month_dir_name)

    ensure_dir(month_dir)
    return month_dir


def save_mail_as_msg(mail_item, base_save_root: str):
    """
    Save a MailItem as .msg inside a Year\MM-Month folder determined from subject.

    Example:
        Subject: "Invoice - October 2025"
        base_save_root: R:\EmailArchive
        → R:\EmailArchive\2025\10-October\YYYYMMDD_HHMMSS_Subject.msg

    If month/year not found in subject, saves under base_save_root\_UnknownPeriod
    """
    target_dir = get_period_folder(base_save_root, mail_item.Subject)

    dt = mail_item.SentOn or mail_item.ReceivedTime
    if isinstance(dt, datetime):
        ts = dt.strftime("%Y%m%d_%H%M%S")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    subject_part = sanitize_filename(mail_item.Subject or "No subject")
    filename = f"{ts}_{subject_part}.msg"

    full_path = os.path.join(target_dir, filename)
    mail_item.SaveAs(full_path, constants.olMSG)


def load_rules_from_excel():
    """Load enabled rules from Excel into a list of dicts."""
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
            logging.warning(f"Skipping rule '{rule_name}' — SaveRoot is blank.")
            continue

        source_folder_parts = [
            p.strip() for p in source_folder_path_raw.split("\\") if p.strip()
        ]
        target_folder_parts = [
            p.strip() for p in target_folder_path_raw.split("\\") if p.strip()
        ]

        if not source_mailbox or not source_folder_parts or not sender_email:
            logging.warning(f"Skipping rule '{rule_name}' — missing required fields.")
            continue

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


def sync_outlook(ns, wait_seconds: int = 30):
    """
    Kick off Outlook Send/Receive for all sync groups and wait a bit.
    Relies on Cached Exchange Mode + Download shared folders.
    """
    try:
        sync_objects = ns.SyncObjects
        count = sync_objects.Count
        if count == 0:
            logging.info("No SyncObjects (send/receive groups) defined. Skipping manual sync.")
            return

        logging.info(f"Triggering Outlook sync for {count} send/receive group(s)...")
        for i in range(1, count + 1):
            so = sync_objects.Item(i)
            try:
                logging.info(f"  → Starting sync group: {so.Name}")
                so.Start()
            except Exception as e:
                logging.warning(f"  ✖ Failed to start sync for group '{so.Name}': {e}")

        logging.info(f"Waiting {wait_seconds} second(s) for sync to complete...")
        time.sleep(wait_seconds)

    except Exception as e:
        logging.warning(f"Could not trigger Outlook sync: {e}")


def check_outlook_connection(ns, rule: dict,
                             max_attempts: int = 3,
                             wait_seconds: int = 10) -> bool:
    """
    Basic connectivity check using the first rule's source folder.
    Tries to read Items.Count a few times. Returns True if successful.
    """
    mailbox = rule["source_mailbox"]
    folder_parts = rule["source_folder_parts"]

    for attempt in range(1, max_attempts + 1):
        try:
            logging.info(
                f"Connection check attempt {attempt}/{max_attempts} "
                f"using mailbox '{mailbox}' folder '\\{'\\'.join(folder_parts)}'"
            )
            src_root = get_mailbox_root(ns, mailbox)
            src_folder = get_folder(src_root, folder_parts)
            _ = src_folder.Items.Count  # just touch it
            logging.info("Outlook connection check passed.")
            return True
        except Exception as e:
            logging.warning(
                f"Outlook connection check failed on attempt {attempt}/{max_attempts}: {e}"
            )
            if attempt < max_attempts:
                time.sleep(wait_seconds)

    logging.error("Outlook connection check failed after all retries. Aborting run.")
    return False


def process_rule(ns, rule: dict) -> int:
    """
    Process one rule; return number of moved emails (excluding today).
    Now filters in Python instead of using Items.Restrict.
    Also “opens” each mail (reads Subject + Body) before moving.
    """
    rule_name = rule["rule_name"]
    source_mailbox = rule["source_mailbox"]
    source_folder_parts = rule["source_folder_parts"]
    sender_email = rule["sender_email"].lower()
    target_mailbox = rule["target_mailbox"]
    target_folder_parts = rule["target_folder_parts"]
    save_root = rule["save_root"]

    logging.info(f"=== Rule: {rule_name} ===")
    logging.info(f"  Source: {source_mailbox}\\{'\\'.join(source_folder_parts)}")
    logging.info(f"  Filter: Sender = {sender_email}")
    logging.info(f"  Target: {target_mailbox}\\{'\\'.join(target_folder_parts)}")
    logging.info(f"  Save base path: {save_root}")

    # Build cutoff = today at 00:00, local time
    now = datetime.now()
    today_start = datetime(now.year, now.month, now.day, 0, 0, 0)

    try:
        src_root = get_mailbox_root(ns, source_mailbox)
        src_folder = get_folder(src_root, source_folder_parts)

        tgt_root = get_mailbox_root(ns, target_mailbox)
        tgt_folder = get_folder(tgt_root, target_folder_parts)
    except Exception as e:
        logging.error(f"  ✖ ERROR locating folders: {e}")
        return 0

    # Snapshot items into a Python list (no live COM indexing)
    try:
        src_items = src_folder.Items
        items_list = [item for item in src_items]
    except Exception as e:
        logging.error(f"  ✖ ERROR reading items from source folder: {e}")
        return 0

    logging.info(f"  → Source folder contains {len(items_list)} item(s) in total")

    moved_count = 0
    scanned_count = 0

    for item in items_list:
        if item is None:
            continue
        if getattr(item, "Class", None) != OL_MAILITEM_CLASS:
            continue

        scanned_count += 1

        try:
            item_sender = (getattr(item, "SenderEmailAddress", "") or "").lower()
            received_time = getattr(item, "ReceivedTime", None)
        except Exception as e:
            logging.warning(f"   ✖ ERROR reading basic properties on item: {e}")
            continue

        # Filter by sender
        if item_sender != sender_email:
            continue

        # Filter by date (exclude today)
        if not isinstance(received_time, datetime):
            # If weird type, skip to be safe
            continue
        if received_time >= today_start:
            continue

        # Force-load the item by accessing Subject and Body
        try:
            _ = item.Subject
            _ = item.Body
        except Exception as e:
            logging.warning(f"   ✖ ERROR loading item body/subject: {e}")
            continue

        subject_display = sanitize_filename(getattr(item, "Subject", "") or "No subject")

        try:
            moved = item.Move(tgt_folder)
            moved.UnRead = True
            moved.Save()

            # Save based on period in subject
            save_mail_as_msg(moved, save_root)

            logging.info(f"   ✔ Moved & saved: {subject_display}")
            moved_count += 1

        except Exception as e:
            logging.error(f"   ✖ ERROR moving/saving '{subject_display}': {e}")

    logging.info(
        f"  → Scanned {scanned_count} mail item(s) in source folder, "
        f"moved {moved_count} matching item(s)."
    )
    return moved_count


# ========= MAIN =========

def main():
    log_path = setup_logging()
    logging.info("Starting email mover run...")

    try:
        rules = load_rules_from_excel()
    except Exception as e:
        logging.error(f"Failed to load rules from Excel: {e}")
        return

    if not rules:
        logging.warning("No enabled rules found or all rules invalid.")
        return

    logging.info(f"Loaded {len(rules)} enabled rule(s).")

    ns = get_outlook_namespace()

    # Sync Outlook before touching folders
    sync_outlook(ns, wait_seconds=30)

    # Connection check using the first rule
    if not check_outlook_connection(ns, rules[0]):
        logging.error("Aborting run due to failed Outlook connection check.")
        return

    total_moved = 0
    for rule in rules:
        moved = process_rule(ns, rule)
        total_moved += moved

    logging.info(
        f"Run finished. Total moved emails across all rules (excluding today): {total_moved}"
    )
    logging.info(f"Details logged to: {log_path}")


if __name__ == "__main__":
    main()
