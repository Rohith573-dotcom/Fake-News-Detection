"""
generate_sample_dataset.py
==========================
Builds `data/news_dataset.csv`, a labelled corpus of 2,500 news articles so the
whole pipeline runs offline with no Kaggle download required.

IMPORTANT
---------
This corpus is SYNTHETIC.  It is generated from stylistic templates that
imitate the register of genuine wire copy versus the register of fabricated
viral news, so that the code, the figures and the report are reproducible on
any machine.  Before final submission, replace it with the real Kaggle
"Fake and Real News" dataset - `main.py` picks up `data/train.csv`
automatically if you drop it in, and every downstream stage is unchanged.

Design note - why the corpus is deliberately *hard*
---------------------------------------------------
A naive generator that draws real articles only from a "real" word list and
fake articles only from a "fake" word list produces perfectly separable
classes, and every classifier then scores 1.00 - a result that tells you
nothing.  Instead each document is given a hidden leaning theta ~ Beta(0.7,0.7);
sentences are drawn from the fabricated pool with probability theta, from the
factual pool with probability 1 - theta, and roughly a third of every document
comes from a shared neutral pool.  The label is simply 1[theta > 0.5], so
documents whose theta sits near 0.5 are honestly ambiguous - exactly the
regime in which the four algorithms start to behave differently.

Usage
-----
    python generate_sample_dataset.py

Author : IICT Summer Internship 2026 (AI & ML)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(2026)

N_ARTICLES   = 2500
LABEL_NOISE  = 0.03      # annotation error, mirrors real crowd-labelled corpora
NEUTRAL_RATE = 0.20      # share of sentences drawn from the shared pool
BETA_A, BETA_B = 0.7, 0.7

# --------------------------------------------------------------------------- #
# Sentence pools
# --------------------------------------------------------------------------- #
FACTUAL = [
    "The measure will take effect from the start of the next financial quarter.",
    "Officials said the revised figures were consistent with earlier projections.",
    "A spokesperson declined to comment on the timeline for implementation.",
    "The report was reviewed by an independent committee before publication.",
    "Analysts noted the change is broadly in line with regional trends.",
    "The department said further details would be shared in due course.",
    "Data for the period was collected across twelve districts.",
    "The proposal is expected to be tabled in the next session.",
    "Preliminary estimates may be revised once the full survey concludes.",
    "The agency added that compliance would be monitored quarterly.",
    "Participants in the study were followed for a period of eighteen months.",
    "The findings were published in a peer reviewed journal on Monday.",
    "According to the notification, existing licences remain valid.",
    "The committee recommended a phased rollout beginning in urban centres.",
    "A senior official said the process would remain open to public comment.",
    "The ministry confirmed that the tender documents are available online.",
    "Researchers cautioned that the sample size limits how far results generalise.",
    "The regulator has invited written objections over the next thirty days.",
    "Year on year growth was recorded at a moderate pace, the bureau said.",
    "The court has listed the matter for a further hearing next month.",
    "Field staff verified the returns before the totals were finalised.",
    "The scheme will be funded through existing budgetary allocations.",
]

FABRICATED = [
    "Share this before it gets deleted forever, they are already removing copies.",
    "An anonymous insider confirmed everything the elites have been hiding.",
    "Nobody in the establishment wants this simple truth to reach ordinary people.",
    "This one weird trick completely destroys the official narrative overnight.",
    "Wake up, the entire system has been lying to you for decades.",
    "Every single expert has been paid off to keep quiet about it.",
    "The evidence is undeniable and it will shock you to your core.",
    "They deleted the original post within minutes, but we saved a copy.",
    "Millions have already shared this, why is nobody in power talking about it.",
    "The truth is finally out and the cover up is collapsing right now.",
    "Doctors hate this because it threatens a billion dollar industry.",
    "Forward this to everyone you love before the censors take it down.",
    "A whistleblower risked everything to leak these explosive documents.",
    "The government has been secretly planning this all along, sources say.",
    "This is the story they tried to bury, and now it is going viral.",
    "What happened next left the entire panel completely speechless.",
    "They have been hiding this in plain sight and calling it policy.",
    "Mainstream outlets refuse to touch this because of who funds them.",
    "You will never look at the official numbers the same way again.",
    "The footage was scrubbed from every platform within the hour.",
    "Insiders are now scrambling to explain the missing paperwork.",
    "Nothing about the official timeline adds up once you read the fine print.",
]

# Shared vocabulary - appears in BOTH classes and dilutes the signal.
NEUTRAL = [
    "The story has attracted considerable attention on social media this week.",
    "Several readers wrote in asking for clarification on the figures.",
    "The issue has been debated at length over the past few months.",
    "It remains unclear how many households will ultimately be affected.",
    "Reactions have been mixed across different parts of the country.",
    "The topic came up again during a televised discussion on Thursday.",
    "Some observers believe the situation will change before the year ends.",
    "The numbers quoted vary depending on which source you consult.",
    "There has been no formal response so far to repeated queries.",
    "The development follows a similar announcement made last year.",
    "Local groups have organised meetings to discuss what happens next.",
    "The full text of the document runs to more than sixty pages.",
]

FACTUAL_HEADS = [
    "Ministry releases updated {t} figures for the quarter",
    "Committee reviews {t} policy after public consultation",
    "Survey finds steady change in {t} indicators across districts",
    "Regulator issues revised guidelines on {t}",
    "Study examines long term effects of {t} programme",
    "Officials outline phased plan for {t} reform",
    "Bureau publishes annual {t} statistics",
    "Court seeks response on {t} petition",
]
FABRICATED_HEADS = [
    "SHOCKING TRUTH ABOUT {T} FINALLY EXPOSED",
    "THEY DON'T WANT YOU TO KNOW THIS ABOUT {T}",
    "BREAKING: THE {T} COVER UP IS REAL",
    "LEAKED DOCUMENT DESTROYS THE OFFICIAL {T} STORY",
    "URGENT WARNING FOR EVERY CITIZEN ABOUT {T}",
    "SCIENTISTS SILENCED AFTER THIS {T} DISCOVERY",
    "THE SECRET {T} AGENDA REVEALED AT LAST",
    "YOU WON'T BELIEVE WHAT THEY DID WITH {T}",
]
NEUTRAL_HEADS = [
    "New questions raised over {t} decision",
    "What the latest {t} report actually says",
    "Debate continues over the future of {t}",
    "A closer look at the {t} numbers",
]

TOPICS = ["the economy", "public health", "the elections", "climate policy",
          "education", "technology", "transport", "agriculture",
          "employment", "housing", "energy prices", "water supply"]


def _headline(theta: float, topic: str) -> str:
    """Pick a headline register that tracks the document's hidden leaning."""
    if RNG.random() < 0.22:
        return RNG.choice(NEUTRAL_HEADS).format(t=topic)
    if RNG.random() < theta:
        return RNG.choice(FABRICATED_HEADS).format(T=topic.upper())
    return RNG.choice(FACTUAL_HEADS).format(t=topic)


def _body(theta: float, n_sent: int) -> str:
    """Sample sentences from the fabricated / factual / neutral pools."""
    out = []
    for _ in range(n_sent):
        if RNG.random() < NEUTRAL_RATE:
            out.append(RNG.choice(NEUTRAL))
        elif RNG.random() < theta:
            out.append(RNG.choice(FABRICATED))
        else:
            out.append(RNG.choice(FACTUAL))
    return " ".join(out)


def build() -> pd.DataFrame:
    rows = []
    for i in range(N_ARTICLES):
        theta = float(RNG.beta(BETA_A, BETA_B))     # hidden leaning
        true_label = int(theta > 0.5)               # 0 = REAL, 1 = FAKE
        topic = str(RNG.choice(TOPICS))

        title = _headline(theta, topic)
        body = _body(theta, int(RNG.integers(5, 14)))

        observed = true_label
        if RNG.random() < LABEL_NOISE:
            observed = 1 - true_label

        rows.append({
            "id": i,
            "title": title,
            "text": f"{title}. {body}",
            "subject": topic,
            "label": observed,
        })

    return pd.DataFrame(rows).sample(frac=1.0, random_state=7).reset_index(drop=True)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = build()
    out = os.path.join("data", "news_dataset.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(f"  rows        : {len(df)}")
    print(f"  REAL (0)    : {(df.label == 0).sum()}")
    print(f"  FAKE (1)    : {(df.label == 1).sum()}")
    print(f"  mean length : {df.text.str.split().str.len().mean():.1f} words")
