import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_curve
import re
from openai import OpenAI
import os
from ddgs import DDGS

model_path = "./fake_news_model"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model.to(device)

DATELINE_RE = re.compile(
    r"^\s*[A-Z][A-Za-z.,\s]{0,40}\((Reuters|AP|AFP)\)\s*-\s*", flags=re.MULTILINE
)

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = DATELINE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_news(text):

    text = clean_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    labels = ["Real", "Fake"]


    print("\nPrediction:", labels[prediction])

    print(
        "Confidence:",
        probabilities[0][prediction].item()
    ) 

def fact_check(article):
    claim_response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {"role": "system", "content": "Extract the single main factual claim from this article as a short search query (under 15 words). Reply with only the query, nothing else."},
            {"role": "user", "content": article},
        ],
    )
    claim = claim_response.choices[0].message.content.strip()

    with DDGS() as ddgs:
        results = list(ddgs.text(claim, max_results=5))

    if not results:
        return f"Claim: {claim}\nVerdict: Uncertain (no search results found)"

    sources_text = "\n\n".join(
        f"Source: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
        for r in results
    )

    verdict_response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "system",
                "content": """You are a fact-checking assistant.
You will be given a claim and real search results about it.
Using ONLY the provided sources (not your own knowledge), determine if the claim is:
Supported, Contradicted, or Uncertain (if sources are insufficient or conflicting).
Cite which source(s) you relied on. Do not invent sources or facts not present above.""",
            },
            {
                "role": "user",
                "content": f"Claim: {claim}\n\nSearch results:\n{sources_text}",
            },
        ],
    )

    return f"Claim: {claim}\n\n{verdict_response.choices[0].message.content}"


article1 = """
Apple held a product event on September 9, 2026, where it introduced its first foldable iPhone, called the iPhone Duo, alongside updated iPhone 18 Pro models, new Apple Watches, and AirPods 5. The foldable model starts around $2,000.
"""

article2 = 'BREAKING: A newly leaked internal memo reveals that a major soft drink company plans to add a secret addictive chemical to its products starting next year, according to anonymous sources close to the company. Health experts are reportedly "extremely concerned" but have declined to comment publicly out of fear of retaliation.'

print(fact_check(article1))

predict_news('Apple held a product event on September 9, 2026, where it introduced its first foldable iPhone, called the iPhone Duo, alongside updated iPhone 18 Pro models, new Apple Watches, and AirPods 5. The foldable model starts around $2,000.')

predict_news('BREAKING: A newly leaked internal memo reveals that a major soft drink company plans to add a secret addictive chemical to its products starting next year, according to anonymous sources close to the company. Health experts are reportedly "extremely concerned" but have declined to comment publicly out of fear of retaliation.')

print(fact_check(article2))
