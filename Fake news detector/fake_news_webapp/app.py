import os
import re

import torch
from flask import Flask, jsonify, render_template, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)

MODEL_DIR = "./fake_news_model"
MAX_LENGTH = 256

DATELINE_RE = re.compile(
    r"^\s*[A-Z][A-Za-z.,\s]{0,40}\((Reuters|AP|AFP)\)\s*-\s*", flags=re.MULTILINE
)


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = DATELINE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Load the RoBERTa classifier once, at startup.
# ---------------------------------------------------------------------------
print("Loading classifier model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()
print("Classifier ready.")


# ---------------------------------------------------------------------------
# Fact-checking setup. Optional: the app still runs (classification only)
# if ddgs/openai aren't installed or OPENROUTER_API_KEY isn't set.
# ---------------------------------------------------------------------------
FACT_CHECK_ENABLED = True
try:
    from ddgs import DDGS
    from openai import OpenAI

    llm_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )
    if not os.environ.get("OPENROUTER_API_KEY"):
        FACT_CHECK_ENABLED = False
except ImportError:
    FACT_CHECK_ENABLED = False


def classify(text: str):
    inputs = tokenizer(
        clean_text(text), return_tensors="pt", truncation=True, max_length=MAX_LENGTH
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[0]
    prediction = torch.argmax(probs).item()
    labels = ["Real", "Fake"]

    return {
        "label": labels[prediction],
        "confidence": round(probs[prediction].item() * 100, 1),
    }


def fact_check(text: str):
    if not FACT_CHECK_ENABLED:
        return {
            "claim": None,
            "verdict": (
                "Fact-checking isn't configured. Install `ddgs` and `openai`, "
                "and set the OPENROUTER_API_KEY environment variable."
            ),
            "sources": [],
        }

    try:
        # Step 1: turn the article into a short, search-friendly claim.
        claim_response = llm_client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract the single main factual claim from this article "
                        "as a short search query (under 15 words). Reply with only "
                        "the query, nothing else."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        claim = claim_response.choices[0].message.content.strip()

        # Step 2: search the web for real, current evidence about that claim.
        with DDGS() as ddgs:
            results = list(ddgs.text(claim, max_results=5))

        if not results:
            return {
                "claim": claim,
                "verdict": "Uncertain — no search results were found for this claim.",
                "sources": [],
            }

        sources_text = "\n\n".join(
            f"Source: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}"
            for r in results
        )

        # Step 3: verdict, grounded only in the retrieved sources.
        verdict_response = llm_client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fact-checking assistant. You will be given a claim "
                        "and real search results about it. Using ONLY the provided "
                        "sources (not your own knowledge), determine if the claim is "
                        "Supported, Contradicted, or Uncertain (if sources are "
                        "insufficient or conflicting). Cite which source(s) you relied "
                        "on. Do not invent sources or facts not present above."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Claim: {claim}\n\nSearch results:\n{sources_text}",
                },
            ],
        )

        return {
            "claim": claim,
            "verdict": verdict_response.choices[0].message.content.strip(),
            "sources": [
                {"title": r.get("title", ""), "url": r.get("href", "")} for r in results
            ],
        }
    except Exception as exc:
        # Never let a search/LLM hiccup take down the whole request.
        return {"claim": None, "verdict": f"Fact-check failed: {exc}", "sources": []}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "Paste an article first."}), 400

    return jsonify(
        {
            "classification": classify(text),
            "fact_check": fact_check(text),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
