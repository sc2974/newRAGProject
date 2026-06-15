import json
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
CORPUS_PATH = BACKEND_DIR / "eval_data" / "corpora" / "legal_clauses_corpus.md"
EVAL_PATH = BACKEND_DIR / "eval_data" / "legal" / "legal_clauses_eval.jsonl"


CLAUSES = [
    ("legal_0001", "Article I Section 1", "All legislative Powers herein granted shall be vested in a Congress of the United States, which shall consist of a Senate and House of Representatives."),
    ("legal_0002", "Article I Section 2", "The House of Representatives shall be composed of Members chosen every second Year by the People of the several States."),
    ("legal_0003", "Article I Section 3", "The Senate of the United States shall be composed of two Senators from each State, chosen for six Years; and each Senator shall have one Vote."),
    ("legal_0004", "Article I Section 7", "Every Bill which shall have passed the House of Representatives and the Senate shall, before it become a Law, be presented to the President of the United States."),
    ("legal_0005", "Article I Section 8 Commerce Clause", "The Congress shall have Power to regulate Commerce with foreign Nations, and among the several States, and with the Indian Tribes."),
    ("legal_0006", "Article I Section 8 Necessary and Proper Clause", "The Congress shall have Power to make all Laws which shall be necessary and proper for carrying into Execution the foregoing Powers."),
    ("legal_0007", "Article I Section 9 Habeas Corpus", "The Privilege of the Writ of Habeas Corpus shall not be suspended, unless when in Cases of Rebellion or Invasion the public Safety may require it."),
    ("legal_0008", "Article II Section 1", "The executive Power shall be vested in a President of the United States of America."),
    ("legal_0009", "Article II Section 2 Commander in Chief", "The President shall be Commander in Chief of the Army and Navy of the United States."),
    ("legal_0010", "Article II Section 2 Treaty Power", "The President shall have Power, by and with the Advice and Consent of the Senate, to make Treaties, provided two thirds of the Senators present concur."),
    ("legal_0011", "Article III Section 1", "The judicial Power of the United States shall be vested in one supreme Court, and in such inferior Courts as the Congress may establish."),
    ("legal_0012", "Article III Section 2", "The judicial Power shall extend to all Cases, in Law and Equity, arising under this Constitution, the Laws of the United States, and Treaties made."),
    ("legal_0013", "Article IV Full Faith and Credit", "Full Faith and Credit shall be given in each State to the public Acts, Records, and judicial Proceedings of every other State."),
    ("legal_0014", "Article IV Privileges and Immunities", "The Citizens of each State shall be entitled to all Privileges and Immunities of Citizens in the several States."),
    ("legal_0015", "Article VI Supremacy Clause", "This Constitution, and the Laws of the United States which shall be made in Pursuance thereof, shall be the supreme Law of the Land."),
    ("legal_0016", "First Amendment", "Congress shall make no law respecting an establishment of religion, or prohibiting the free exercise thereof; or abridging the freedom of speech, or of the press."),
    ("legal_0017", "Second Amendment", "A well regulated Militia, being necessary to the security of a free State, the right of the people to keep and bear Arms shall not be infringed."),
    ("legal_0018", "Fourth Amendment", "The right of the people to be secure in their persons, houses, papers, and effects, against unreasonable searches and seizures, shall not be violated."),
    ("legal_0019", "Fifth Amendment Due Process", "No person shall be deprived of life, liberty, or property, without due process of law."),
    ("legal_0020", "Fifth Amendment Self Incrimination", "No person shall be compelled in any criminal case to be a witness against himself."),
    ("legal_0021", "Sixth Amendment", "In all criminal prosecutions, the accused shall enjoy the right to a speedy and public trial, by an impartial jury."),
    ("legal_0022", "Eighth Amendment", "Excessive bail shall not be required, nor excessive fines imposed, nor cruel and unusual punishments inflicted."),
    ("legal_0023", "Tenth Amendment", "The powers not delegated to the United States by the Constitution, nor prohibited by it to the States, are reserved to the States respectively, or to the people."),
    ("legal_0024", "Fourteenth Amendment Equal Protection", "No State shall deny to any person within its jurisdiction the equal protection of the laws."),
]


QUESTIONS = [
    ("Which body receives legislative powers granted by the Constitution?", "All legislative Powers herein granted shall be vested in a Congress"),
    ("How often are members of the House of Representatives chosen?", "chosen every second Year"),
    ("How many senators does each state have?", "two Senators from each State"),
    ("What must happen before a bill becomes law?", "presented to the President of the United States"),
    ("Which clause lets Congress regulate commerce among the states?", "regulate Commerce with foreign Nations, and among the several States"),
    ("What clause lets Congress pass laws needed to execute its powers?", "necessary and proper"),
    ("When may habeas corpus be suspended?", "Rebellion or Invasion"),
    ("Where is executive power vested?", "vested in a President"),
    ("Who is Commander in Chief of the Army and Navy?", "The President shall be Commander in Chief"),
    ("What Senate approval is needed for treaties?", "two thirds of the Senators present concur"),
    ("Where is judicial power vested?", "one supreme Court"),
    ("What kinds of cases does federal judicial power cover?", "arising under this Constitution"),
    ("What must states give to other states' public acts and records?", "Full Faith and Credit"),
    ("What protection do citizens receive across states?", "Privileges and Immunities"),
    ("What is the supreme law of the land?", "supreme Law of the Land"),
    ("Which amendment protects freedom of speech and press?", "freedom of speech, or of the press"),
    ("Which amendment protects the right to keep and bear arms?", "keep and bear Arms"),
    ("Which amendment protects against unreasonable searches and seizures?", "unreasonable searches and seizures"),
    ("What process is required before depriving a person of life, liberty, or property?", "due process of law"),
    ("Which amendment protects against compelled self-incrimination?", "witness against himself"),
    ("What trial rights are guaranteed in criminal prosecutions?", "speedy and public trial"),
    ("Which amendment bans cruel and unusual punishments?", "cruel and unusual punishments"),
    ("Who keeps powers not delegated to the United States?", "reserved to the States respectively, or to the people"),
    ("What protection does the Fourteenth Amendment give equally under law?", "equal protection of the laws"),
    ("What does Article I Section 7 require for bills?", "before it become a Law, be presented to the President"),
    ("What does the Commerce Clause allow Congress to regulate?", "Commerce with foreign Nations"),
    ("What is required for the President to make treaties?", "Advice and Consent of the Senate"),
    ("Which clause makes federal law supreme over state law?", "supreme Law of the Land"),
    ("Which clause concerns records and judicial proceedings of other states?", "public Acts, Records, and judicial Proceedings"),
    ("What does the Tenth Amendment reserve?", "powers not delegated to the United States"),
]


def main() -> None:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_corpus()
    write_eval()
    print(json.dumps({"clauses": len(CLAUSES), "questions": len(QUESTIONS), "corpus": str(CORPUS_PATH), "eval": str(EVAL_PATH)}, indent=2))


def write_corpus() -> None:
    with CORPUS_PATH.open("w", encoding="utf-8", newline="\n") as file:
        for doc_id, title, text in CLAUSES:
            file.write(f"# DOCID:{doc_id} | {title}\n\n")
            file.write(f"{title}\n")
            file.write(f"{text}\n\n")


def write_eval() -> None:
    with EVAL_PATH.open("w", encoding="utf-8", newline="\n") as file:
        for index, (query, answer) in enumerate(QUESTIONS, start=1):
            row = {
                "id": f"legal_{index:04d}",
                "query": query,
                "answers": [answer],
                "expected_document": "legal_clauses_corpus.md",
                "tags": ["en", "legal", "clause"],
                "group_id": "legal_clauses",
                "context_id": "legal_clauses",
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
