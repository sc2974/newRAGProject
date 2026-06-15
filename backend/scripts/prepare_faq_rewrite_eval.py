import json
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
SOURCE = BACKEND_DIR / "eval_data" / "faq" / "customer_support_faq_eval.jsonl"
OUTPUT = BACKEND_DIR / "eval_data" / "faq" / "customer_support_faq_rewrite_eval.jsonl"


REWRITES = {
    "How can I create an account?": "What do I need to do to sign up?",
    "What payment methods do you accept?": "Which ways can I pay for my purchase?",
    "How can I track my order?": "Where can I see the shipping status of my order?",
    "What is your return policy?": "How do returns and refunds work?",
    "Can I cancel my order?": "Is it possible to stop an order before it ships?",
    "How long does shipping take?": "When should I expect my delivery to arrive?",
    "Do you offer international shipping?": "Can customers outside the country receive orders?",
    "What should I do if my package is lost or damaged?": "Who should I contact about a missing or broken delivery?",
    "Can I change my shipping address after placing an order?": "How can I update the delivery address after checkout?",
    "How can I contact customer support?": "What is the best way to reach your support team?",
    "Do you offer gift wrapping services?": "Can my order be wrapped as a gift?",
    "What is your price matching policy?": "Will you match a lower competitor price?",
    "Can I order by phone?": "Do you allow purchases over a phone call?",
    "Are my personal and payment details secure?": "Is my private and card information protected?",
    "What is your price adjustment policy?": "Can I get money back if the item goes on sale after I buy it?",
    "Do you have a loyalty program?": "Can I earn rewards for repeat purchases?",
    "Can I order without creating an account?": "Is guest checkout available?",
    "Do you offer bulk or wholesale discounts?": "Can I get a discount for buying a large quantity?",
    "Can I change or cancel an item in my order?": "How do I modify or remove something from an existing order?",
    "How can I leave a product review?": "Where do I submit feedback about a product I bought?",
}


def main() -> None:
    rewritten = []
    with SOURCE.open("r", encoding="utf-8") as source_file:
        for line in source_file:
            row = json.loads(line)
            original_query = row["query"]
            row["original_query"] = original_query
            row["query"] = REWRITES.get(original_query, fallback_rewrite(original_query))
            row["id"] = f"{row['id']}_rewrite"
            row["tags"] = list(dict.fromkeys(row.get("tags", []) + ["rewritten_query"]))
            rewritten.append(row)

    with OUTPUT.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in rewritten:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"rows": len(rewritten), "output": str(OUTPUT)}, indent=2))


def fallback_rewrite(query: str) -> str:
    replacements = [
        ("How can I", "What is the way to"),
        ("How do I", "What steps should I take to"),
        ("Can I", "Is it possible to"),
        ("Do you", "Does your store"),
        ("What is", "Can you explain"),
        ("What are", "Can you list"),
    ]
    for source, target in replacements:
        if query.startswith(source):
            return query.replace(source, target, 1)
    return f"Please help me with this issue: {query}"


if __name__ == "__main__":
    main()
