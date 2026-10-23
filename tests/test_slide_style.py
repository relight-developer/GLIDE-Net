"""Copy-style checks for the generated deck.

``tools/check_slides.py`` runs three families of assertions on the exported
package. Layout QA needs the pptx itself, but the wording rules are pure string
work, so they can be tested here without the binary and without PowerPoint.

The rules come from the house style used for the deck: no listing colons, no
em dashes, and no "是 A 而不是 B" reversal sentences.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.check_slides import style_problems  # noqa: E402


def test_listing_colon_is_rejected():
    assert style_problems("分成五块：Adapter、CNN、CMF") == ["禁用标点『：』"]


def test_em_dash_is_rejected():
    assert style_problems("端到端 60.9 ms —— 362 张实测")


def test_turncoat_sentence_is_rejected():
    assert style_problems("这不是加速手段，而是可移植性保证") == ["翻案句『不是…而是…』"]


def test_plain_copy_passes():
    assert style_problems("分割找像素，检测认类别，规则定档位") == []
