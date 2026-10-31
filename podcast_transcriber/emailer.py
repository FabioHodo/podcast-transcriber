"""Send the email via Resend (send-only API key, preferred) or plain SMTP."""

from __future__ import annotations

import base64
import json
import logging
import smtplib
import ssl
import urllib.error
import urllib.request
from email.message import EmailMessage
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)

RESEND_URL = "https://api.resend.com/emails"
# Resend's shared test sender. Without a verified domain it can only deliver to the
# address you signed up to Resend with, which is exactly what this project needs.
RESEND_DEFAULT_FROM = "Podcast Transcriber <onboarding@resend.dev>"


def send_email(cfg: Config, to: str, subject: str, plain: str, html: str,
               attachment: Path | None) -> None:
    """Send one message to one recipient. Each person gets their own copy, so nobody
    on the list sees anyone else's address."""
    if cfg.resend_api_key:
        send_resend(cfg, resend_payload(cfg, to, subject, plain, html, attachment))
    else:
        send_smtp(cfg, build_message(cfg, to, subject, plain, html, attachment))


# --- Resend ---------------------------------------------------------------------

def resend_payload(cfg: Config, to: str, subject: str, plain: str, html: str,
                   attachment: Path | None) -> dict:
    payload = {
        "from": cfg.email_from or RESEND_DEFAULT_FROM,
        "to": [to],
        "subject": subject,
        "text": plain,
        "html": html,
    }
    if attachment is not None:
        payload["attachments"] = [{
            "filename": attachment.name,
            "content": base64.b64encode(attachment.read_bytes()).decode("ascii"),
        }]
    return payload


def send_resend(cfg: Config, payload: dict) -> None:
    req = urllib.request.Request(
        RESEND_URL,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {cfg.resend_api_key}",
            "Content-Type": "application/json",
            "User-Agent": "podcast-transcriber/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        hint = ""
        if exc.code in (403, 422) and "onboarding@resend.dev" in payload["from"]:
            hint = (" (Resend's test sender only delivers to the address you signed up with. "
                    "To email anyone else, verify a domain in Resend and set EMAIL_FROM, "
                    "e.g. 'All-In Transcripts <transcripts@yourdomain.com>'.)")
        raise RuntimeError(f"Resend error {exc.code}: {detail}{hint}") from None
    log.debug("Resend accepted message %s", body.get("id"))


# --- SMTP -----------------------------------------------------------------------

def build_message(
    cfg: Config, to: str, subject: str, plain: str, html: str, attachment: Path | None
) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.email_from or cfg.smtp_user or ""
    msg["To"] = to
    msg.set_content(plain)
    msg.add_alternative(html, subtype="html")
    if attachment is not None:
        msg.add_attachment(
            attachment.read_bytes(),
            maintype="text",
            subtype="plain",
            filename=attachment.name,
        )
    return msg


def send_smtp(cfg: Config, msg: EmailMessage) -> None:
    if not (cfg.smtp_user and cfg.smtp_password):
        raise RuntimeError("Set RESEND_API_KEY, or SMTP_USER and SMTP_PASSWORD, to send email")
    context = ssl.create_default_context()
    if cfg.smtp_port == 465:
        with smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, context=context, timeout=60) as smtp:
            smtp.login(cfg.smtp_user, cfg.smtp_password)
            smtp.send_message(msg)
    else:
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=60) as smtp:
            smtp.starttls(context=context)
            smtp.login(cfg.smtp_user, cfg.smtp_password)
            smtp.send_message(msg)
