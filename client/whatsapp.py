"""
client/whatsapp.py — WhatsApp Cloud API (Meta Graph API v18) client.

Handles:
  - Getting media download URL from a media_id
  - Downloading audio bytes from Meta's CDN
  - Sending text replies back to the sender

Verdict message templates follow WhatsApp Business Policy:
  - Every reply includes opt-out footer ("Reply STOP")
  - Fact-checker link included in every reply
  - Electoral disclaimer can be injected per-message

All requests use httpx (sync) since Celery tasks are synchronous.
"""

import httpx
import structlog

from utils.exceptions import AudioDownloadError

log = structlog.get_logger(__name__)

# ── Verdict message templates ─────────────────────────────────────────────────
VERDICTS = {
    "fake": (
        "🚨 *POSSIBLE FAKE AUDIO*\n\n"
        "This audio shows strong signs of digital manipulation.\n"
        "Score: {pct}%\n\n"
        "Please do NOT forward this clip. Share this result instead."
    ),
    "suspicious": (
        "⚠️ *SUSPICIOUS AUDIO*\n\n"
        "This clip has unusual patterns. It may be edited.\n"
        "Score: {pct}%\n\n"
        "Wait for more information before sharing."
    ),
    "real": (
        "✅ *AUDIO APPEARS REAL*\n\n"
        "No clear signs of manipulation found.\n"
        "Score: {pct}%"
    ),
}

FOOTER = (
    "\n\n_Reply STOP to unsubscribe. Automated analysis only.\n"
    "Visit {fact_checker} for verification._"
)

ELECTORAL_DISCLAIMER = (
    "\n\n_This analysis is automated and does not constitute an official fact-check. "
    "The Election Commission of India does not endorse or operate this service._"
)

FIRST_REPLY_PRIVACY_NOTICE = (
    "\n\n_Privacy notice: We do not store your phone number. "
    "Audio is deleted after analysis. "
    "Privacy policy: {fact_checker}/privacy_"
)


class WhatsAppClient:
    """Client for the WhatsApp Cloud API (Meta Graph API v18+).

    Args:
        token:          WhatsApp permanent access token (WA_TOKEN env var).
        phone_id:       WhatsApp business phone number ID (PHONE_NUMBER_ID env var).
        fact_checker_url: URL included in every reply footer.
        timeout:        HTTP request timeout in seconds. Default 15.

    Example:
        client = WhatsAppClient(token="...", phone_id="...", fact_checker_url="https://factchecker.in")
        client.send_verdict(to="919876543210", label="fake", confidence_pct=91)
    """

    _BASE = "https://graph.facebook.com/v18.0"

    def __init__(
        self,
        token: str,
        phone_id: str,
        fact_checker_url: str = "https://factchecker.in",
        timeout: int = 15,
    ) -> None:
        self.base = f"{self._BASE}/{phone_id}"
        self.media_base = self._BASE
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self.fact_checker_url = fact_checker_url
        self.timeout = timeout

    def get_media_url(self, media_id: str) -> str:
        """Fetch the temporary CDN download URL for a WhatsApp media object.

        Args:
            media_id: The media ID from the webhook event payload.

        Returns:
            HTTPS download URL (valid for ~5 minutes).

        Raises:
            AudioDownloadError: If the API call fails.
        """
        url = f"{self.media_base}/{media_id}"
        try:
            resp = httpx.get(url, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            download_url: str = resp.json()["url"]
            log.debug("media_url_fetched", media_id=media_id)
            return download_url
        except httpx.HTTPStatusError as exc:
            raise AudioDownloadError(
                f"Failed to fetch media URL for {media_id}: HTTP {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            raise AudioDownloadError(f"Unexpected error fetching media URL: {exc}") from exc

    def download(self, url: str) -> bytes:
        """Download audio bytes from a Meta CDN URL.

        Args:
            url: Temporary download URL obtained from get_media_url().

        Returns:
            Raw audio bytes (ogg/opus format as delivered by WhatsApp).

        Raises:
            AudioDownloadError: If the download fails.
        """
        try:
            resp = httpx.get(url, headers=self.headers, timeout=30, follow_redirects=True)
            resp.raise_for_status()
            log.info("audio_downloaded", bytes=len(resp.content))
            return resp.content
        except httpx.HTTPStatusError as exc:
            raise AudioDownloadError(
                f"Audio download failed: HTTP {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            raise AudioDownloadError(f"Unexpected error downloading audio: {exc}") from exc

    def send_message(self, to: str, text: str) -> httpx.Response:
        """Send a plain text WhatsApp message.

        Args:
            to:   Recipient phone number in E.164 format (e.g. "919876543210").
            text: Message body string.

        Returns:
            httpx.Response from the Meta Graph API.
        """
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": text},
        }
        resp = httpx.post(
            f"{self.base}/messages",
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        if resp.is_error:
            log.error("send_message_failed", to=to[:6] + "***", status=resp.status_code, body=resp.text[:200])
        else:
            log.info("send_message_ok", to=to[:6] + "***", status=resp.status_code)
        return resp

    def send_verdict(
        self,
        to: str,
        label: str,
        confidence_pct: int,
        meta_flags: list[str] | None = None,
        electoral_period: bool = False,
        first_interaction: bool = False,
    ) -> httpx.Response:
        """Send a formatted verdict reply to the user.

        Composes the appropriate verdict template, appends footer with opt-out
        and fact-checker link, and optionally appends the electoral disclaimer.

        Args:
            to:               Recipient phone number (E.164).
            label:            "fake" | "suspicious" | "real".
            confidence_pct:   Integer 0–100.
            meta_flags:       Optional list of heuristic flag strings to include.
            electoral_period: If True, append the election disclaimer.
            first_interaction: If True, include the privacy notice link.

        Returns:
            httpx.Response from the Meta Graph API.
        """
        template = VERDICTS.get(label, VERDICTS["real"])
        body = template.format(pct=confidence_pct)

        # Append metadata heuristic flags if any
        if meta_flags:
            flags_str = ", ".join(meta_flags)
            body += f"\n\n_Technical flags: {flags_str}_"

        # Footer with opt-out and fact-checker link
        body += FOOTER.format(fact_checker=self.fact_checker_url)

        # Electoral disclaimer (mandatory during election periods)
        if electoral_period:
            body += ELECTORAL_DISCLAIMER

        # Privacy notice for first-contact users
        if first_interaction:
            body += FIRST_REPLY_PRIVACY_NOTICE.format(fact_checker=self.fact_checker_url)

        return self.send_message(to, body)

    def send_rate_limit_message(self, to: str) -> httpx.Response:
        """Notify the sender that their daily quota is exhausted.

        Args:
            to: Recipient phone number (E.164).

        Returns:
            httpx.Response from the Meta Graph API.
        """
        text = (
            "You have reached your limit of 10 audio checks per day.\n"
            "Your limit resets at midnight.\n\n"
            "_Reply STOP to unsubscribe._"
        )
        return self.send_message(to, text)

    def send_low_confidence_warning(self, to: str) -> httpx.Response:
        """Warn sender that the audio clip is too short for reliable analysis.

        Args:
            to: Recipient phone number (E.164).

        Returns:
            httpx.Response from the Meta Graph API.
        """
        text = (
            "⚠️ *Audio too short for reliable analysis*\n\n"
            "Clips under 3 seconds produce unreliable results.\n"
            "Please send a longer clip.\n\n"
            f"_Reply STOP to unsubscribe. {self.fact_checker_url}_"
        )
        return self.send_message(to, text)
