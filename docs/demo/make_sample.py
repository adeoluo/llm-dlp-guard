"""Write the demo inputs to docs/demo/.sample/ (not committed).

The person is fictional and the credentials are random strings in real formats, generated here
so the repository itself holds no credential-shaped text.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from llm_dlp_guard import secrets_corpus as sc

OUT = Path(__file__).resolve().parent / ".sample"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = random.Random(2026)
    github = sc.github_token(rng)
    aws = sc.aws_access_key(rng)
    prompt = (
        "Hi, I'm Marta Kowalski from the Lyon office. Please draft a reply to the customer\n"
        "Bram Okafor (bram.okafor@example.org, +44 20 7946 0958) about the refund on\n"
        "card 4111 1111 1111 1111 to IBAN GB82 WEST 1234 5698 7654 32.\n"
        "He lives at 12 Mill Lane, Utrecht.\n"
        f"Our deploy failed with GITHUB_TOKEN={github} and key {aws} from 81.2.69.160.\n"
    )
    (OUT / "prompt.txt").write_text(prompt, encoding="utf-8")
    answer = (
        "Thanks for the details. The build log shows the job read "
        f"GITHUB_TOKEN={github} from the environment. Rotate it, then rerun.\n"
        "I have also drafted the customer email to bram.okafor@example.org; "
        "it confirms the refund\n"
        "to card 4111 1111 1111 1111 and asks him to call +44 20 7946 0958 with any questions.\n"
    )
    (OUT / "answer.txt").write_text(answer, encoding="utf-8")
    chat = {"model": "echo", "messages": [{"role": "user", "content": prompt}]}
    (OUT / "chat.json").write_text(json.dumps(chat), encoding="utf-8")
    (OUT / "chat_stream.json").write_text(json.dumps({**chat, "stream": True}), encoding="utf-8")
    key = sc.private_key(rng)
    blocked = {
        "model": "echo",
        "messages": [{"role": "user", "content": "Install this key:\n" + key}],
    }
    (OUT / "chat_key.json").write_text(json.dumps(blocked), encoding="utf-8")


if __name__ == "__main__":
    main()
