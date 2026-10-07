"""Write the hand-judged E4 (multilingual) and E6 (jurisdiction) sets.  [owner: Kashvi (E4), Gaurav (E6)]

    python scripts/08_make_judged_sets.py          # after make data

E4, one legal need asked three ways (statute retrieval) - 50 needs, 150 queries
    Each need below is written in English, Hindi (Devanagari) and Roman Hinglish, the way a
    person would type it. The judges chose the sections that answer the need; every listed
    section is mapped to its BNS equivalents (via the crosswalk); those BNS sections are judged
    relevant, grade 2 for the first-listed ("primary") offence and 1 for the rest. The incident
    date is 2025, so only the BNS reading is in force: an IPC section is the wrong answer for
    this user (the same rule as E7), even though it describes the same offence.

E6, the same question asked from different states (precedent retrieval)
    Topical relevance = the precedent applied the topic's section (IL-PCSR citation labels).
    run_eval then grades it 2 if the court binds the asking state (SC or that state's High
    Court) and 1 if only persuasive (eval/qrels.jurisdiction_grades).

Review the needs and sections before reporting: they are the team's judgments.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

# need id, sections (primary first), English, Hindi, Hinglish
NEEDS = [
    ("stab", ["ipc:324", "ipc:326", "ipc:307"],
     "my brother was stabbed with a knife in a fight",
     "झगड़े में मेरे भाई को चाकू मारा गया",
     "jhagde mein mere bhai ko chaku maara"),
    ("dowry_cruelty", ["ipc:498a"],
     "husband and in-laws harass wife for dowry",
     "पति और ससुराल वाले दहेज के लिए पत्नी को परेशान करते हैं",
     "pati aur sasural wale dahej ke liye patni ko pareshan karte hain"),
    ("dowry_death", ["ipc:304b"],
     "woman died within seven years of marriage after dowry demands",
     "शादी के सात साल के अंदर दहेज की मांग के बाद महिला की मौत",
     "shaadi ke saat saal ke andar dahej ki maang ke baad mahila ki maut"),
    ("theft", ["ipc:379", "ipc:378"],
     "someone stole my mobile phone",
     "किसी ने मेरा मोबाइल फोन चुरा लिया",
     "kisi ne mera mobile phone chura liya"),
    ("cheating", ["ipc:420", "ipc:415"],
     "cheated of money in an online fraud",
     "ऑनलाइन धोखाधड़ी में पैसे ठग लिए",
     "online dhokhadhadi mein paise thag liye"),
    ("threat", ["ipc:506"],
     "neighbour threatened to kill me",
     "पड़ोसी ने मुझे जान से मारने की धमकी दी",
     "padosi ne mujhe jaan se maarne ki dhamki di"),
    ("molestation", ["ipc:354", "ipc:509"],
     "man touched a woman inappropriately and outraged her modesty",
     "आदमी ने महिला को गलत तरीके से छुआ और छेड़छाड़ की",
     "aadmi ne mahila ke saath chhedchhad ki"),
    ("rape", ["ipc:376", "ipc:375"],
     "punishment for rape",
     "बलात्कार की सजा",
     "balatkar ki saza"),
    ("kidnap_child", ["ipc:363", "ipc:361"],
     "child kidnapped from school",
     "स्कूल से बच्चे का अपहरण",
     "school se bachche ka apharan ho gaya"),
    ("burglary", ["ipc:457", "ipc:454", "ipc:380"],
     "house broken into at night and jewellery stolen",
     "रात में घर में सेंध लगाकर गहने चोरी",
     "raat mein ghar mein sendh lagakar gehne chori"),
    ("rash_driving", ["ipc:304a", "ipc:279"],
     "death caused by rash and negligent driving",
     "लापरवाही से गाड़ी चलाने से मौत",
     "laparwahi se gaadi chalane se maut"),
    ("murder", ["ipc:302", "ipc:300"],
     "punishment for murder",
     "हत्या की सजा",
     "hatya ki saza kya hai"),
    ("robbery", ["ipc:392", "ipc:397"],
     "robbed at knifepoint on the road",
     "सड़क पर चाकू दिखाकर लूट",
     "sadak par chaku dikhakar loot liya"),
    ("defamation", ["ipc:499", "ipc:500"],
     "false statements damaging my reputation defamation",
     "झूठी बातें फैलाकर मेरी मानहानि की",
     "jhoothi baatein phailakar meri manhani ki"),
    ("forgery", ["ipc:463", "ipc:465", "ipc:467", "ipc:468", "ipc:471"],
     "forged signature on property documents",
     "संपत्ति के कागजों पर जाली हस्ताक्षर",
     "property ke kagazon par jaali signature"),
    ("breach_of_trust", ["ipc:405", "ipc:406", "ipc:408"],
     "employee misappropriated company money entrusted to him",
     "कर्मचारी ने सौंपा गया पैसा हड़प लिया",
     "karmchari ne saunpa gaya paisa hadap liya"),
    ("abet_suicide", ["ipc:306"],
     "husband drove wife to commit suicide",
     "पति ने पत्नी को आत्महत्या के लिए उकसाया",
     "pati ne patni ko aatmhatya ke liye uksaya"),
    ("extortion", ["ipc:384", "ipc:386"],
     "demanding money by threatening to harm family",
     "परिवार को नुकसान पहुंचाने की धमकी देकर पैसे की वसूली",
     "parivar ko nuksan ki dhamki dekar paise ki vasooli"),
    ("confinement", ["ipc:342"],
     "locked me in a room and did not let me leave",
     "मुझे कमरे में बंद कर दिया और जाने नहीं दिया",
     "mujhe kamre mein band kar diya aur jaane nahi diya"),
    ("bigamy", ["ipc:494"],
     "husband married again while first wife is alive",
     "पहली पत्नी के रहते पति ने दूसरी शादी की",
     "pehli patni ke rehte pati ne doosri shaadi kar li"),
    # --- needs 21-50 (added for the second round; 45-50 use the CrPC->BNSS / IEA->BSA bridge) ---
    ("wrongful_restraint", ["ipc:341"],
     "they blocked my way and did not let me pass",
     "उन्होंने मेरा रास्ता रोक दिया और जाने नहीं दिया",
     "unhone mera rasta rok diya aur jaane nahi diya"),
    ("simple_hurt", ["ipc:323"],
     "neighbour slapped and beat me",
     "पड़ोसी ने मुझे थप्पड़ मारा और पीटा",
     "padosi ne mujhe thappad maara aur peeta"),
    ("grievous_hurt", ["ipc:325"],
     "my arm was broken in a fight",
     "झगड़े में मेरा हाथ तोड़ दिया",
     "jhagde mein mera haath tod diya"),
    ("acid_attack", ["ipc:326a"],
     "acid thrown on a woman's face",
     "महिला के चेहरे पर तेजाब फेंका",
     "mahila ke chehre par tezaab phenka"),
    ("stalking", ["ipc:354d"],
     "a man keeps following a woman and messaging her",
     "एक आदमी लड़की का पीछा करता है और मैसेज करता है",
     "ek aadmi ladki ka peecha karta hai aur message karta hai"),
    ("sexual_harassment", ["ipc:354a"],
     "sexual harassment by a colleague at the office",
     "दफ्तर में सहकर्मी ने यौन उत्पीड़न किया",
     "office mein colleague ne yaun utpeedan kiya"),
    ("trespass", ["ipc:447", "ipc:441"],
     "someone entered my land without permission",
     "किसी ने बिना अनुमति मेरी जमीन में घुसपैठ की",
     "kisi ne bina anumati meri zameen mein ghuspaith ki"),
    ("mischief", ["ipc:427", "ipc:425"],
     "he deliberately damaged my car",
     "उसने जानबूझकर मेरी गाड़ी को नुकसान पहुंचाया",
     "usne jaanbujhkar meri gaadi ko nuksan pahunchaya"),
    ("arson", ["ipc:436", "ipc:435"],
     "they set fire to my house",
     "उन्होंने मेरे घर में आग लगा दी",
     "unhone mere ghar mein aag laga di"),
    ("dacoity", ["ipc:395", "ipc:391"],
     "a gang of armed men looted the house",
     "हथियारबंद गिरोह ने घर लूट लिया",
     "hathiyarband giroh ne ghar loot liya"),
    ("stolen_property", ["ipc:411"],
     "bought a stolen phone knowing it was stolen",
     "जानबूझकर चोरी का फोन खरीदा",
     "jaanbujhkar chori ka phone kharida"),
    ("impersonation", ["ipc:419", "ipc:416"],
     "fraudster pretended to be a bank officer and cheated me",
     "ठग ने बैंक अधिकारी बनकर मुझे धोखा दिया",
     "thag ne bank adhikari bankar mujhe dhokha diya"),
    ("counterfeit_currency", ["ipc:489b", "ipc:489c"],
     "using fake currency notes",
     "नकली नोट चलाना",
     "nakli note chalana"),
    ("rioting", ["ipc:147", "ipc:146"],
     "a mob rioted with sticks",
     "लाठियों के साथ भीड़ ने दंगा किया",
     "lathiyon ke saath bheed ne danga kiya"),
    ("assault_public_servant", ["ipc:353"],
     "assaulted a police officer on duty",
     "ड्यूटी पर पुलिसकर्मी पर हमला किया",
     "duty par policewale par hamla kiya"),
    ("false_evidence", ["ipc:193", "ipc:191"],
     "gave false evidence in court",
     "अदालत में झूठी गवाही दी",
     "adalat mein jhoothi gawahi di"),
    ("conspiracy", ["ipc:120b", "ipc:120a"],
     "they planned the crime together in a conspiracy",
     "उन्होंने मिलकर अपराध की साजिश रची",
     "unhone milkar apradh ki saazish rachi"),
    ("culpable_homicide", ["ipc:304", "ipc:299"],
     "killed him in a sudden fight without intending to murder",
     "अचानक झगड़े में बिना इरादे के उसकी मौत हो गई",
     "achanak jhagde mein bina irade ke uski maut ho gayi"),
    ("kidnap_ransom", ["ipc:364a"],
     "kidnapped a businessman for ransom",
     "फिरौती के लिए व्यापारी का अपहरण",
     "phirauti ke liye vyapari ka apharan"),
    ("trafficking", ["ipc:370"],
     "trafficking of girls for forced work",
     "जबरन काम के लिए लड़कियों की तस्करी",
     "jabran kaam ke liye ladkiyon ki taskari"),
    ("obscenity", ["ipc:294"],
     "obscene acts and songs in a public place",
     "सार्वजनिक स्थान पर अश्लील हरकत और गाने",
     "public jagah par ashleel harkat aur gaane"),
    ("public_nuisance", ["ipc:268"],
     "causing public nuisance on the road",
     "सड़क पर सार्वजनिक उपद्रव",
     "sadak par sarvajanik upadrav"),
    ("medical_negligence", ["ipc:337", "ipc:338"],
     "injury caused by a doctor's negligence",
     "डॉक्टर की लापरवाही से चोट",
     "doctor ki laparwahi se chot"),
    ("religious_insult", ["ipc:295a"],
     "insulting a religion to outrage religious feelings",
     "धर्म का अपमान कर धार्मिक भावनाओं को ठेस पहुंचाना",
     "dharm ka apmaan kar dharmik bhavnaon ko thes pahunchana"),
    ("anticipatory_bail", ["crpc:438"],
     "anticipatory bail before arrest",
     "गिरफ्तारी से पहले अग्रिम जमानत",
     "giraftari se pehle agrim zamanat"),
    ("regular_bail", ["crpc:437", "crpc:439"],
     "bail in a non-bailable offence",
     "गैर जमानती अपराध में जमानत",
     "gair zamanati apradh mein zamanat"),
    ("fir_refused", ["crpc:154"],
     "police refused to register my FIR",
     "पुलिस ने मेरी एफआईआर दर्ज नहीं की",
     "police ne meri FIR darj nahi ki"),
    ("maintenance", ["crpc:125"],
     "wife claims maintenance from her husband",
     "पत्नी पति से भरण पोषण मांगती है",
     "patni pati se bharan poshan maangti hai"),
    ("quash_fir", ["crpc:482"],
     "quash the FIR using the inherent powers of the High Court",
     "उच्च न्यायालय की अंतर्निहित शक्ति से एफआईआर रद्द करवाना",
     "high court ki shakti se FIR radd karwana"),
    ("electronic_evidence", ["iea:65b"],
     "certificate for WhatsApp messages as electronic evidence",
     "व्हाट्सएप संदेशों को इलेक्ट्रॉनिक सबूत के रूप में प्रमाण पत्र",
     "whatsapp message ko electronic saboot ke roop mein certificate"),
]

# topic id, query text, sections that make a precedent topically relevant
TOPICS = [
    ("murder_conviction", "appeal against murder conviction on circumstantial evidence", ["ipc:302"]),
    ("cheque_bounce", "cheque dishonour complaint under negotiable instruments act", ["nia:138"]),
    ("quash_fir", "quashing of FIR under inherent powers of high court", ["crpc:482"]),
    ("cruelty_498a", "cruelty by husband and relatives for dowry", ["ipc:498a"]),
    ("attempt_murder", "attempt to murder conviction and sentence", ["ipc:307"]),
    ("cheating_420", "cheating and dishonestly inducing delivery of property", ["ipc:420"]),
    ("abet_suicide", "abetment of suicide by harassment", ["ipc:306"]),
    ("rape_376", "rape conviction based on testimony of prosecutrix", ["ipc:376"]),
    ("rash_driving", "death by rash and negligent driving", ["ipc:304a"]),
    ("motor_claims", "compensation claim for motor accident before tribunal", ["mva:166"]),
]
STATES = ["delhi", "maharashtra", "punjab", "uttar-pradesh"]


def main() -> None:
    from kanoon_bridge.config import load_config, project_path
    from kanoon_bridge.schema import read_documents
    from kanoon_bridge.text.version_norm import VersionNormalizer

    cfg = load_config()
    norm = VersionNormalizer.load(cfg)
    docs = list(read_documents(project_path(cfg.paths.docs)))
    by_ref: dict[str, list[str]] = {}
    for d in docs:
        if d.doc_type.value == "statute" and (d.meta or {}).get("ref"):
            ref = d.meta["ref"]
            by_ref.setdefault(ref, []).append(d.doc_id)
            base = re.match(r"(\w+:\d+[a-z]*)", ref)
            if base and base.group(1) != ref:
                by_ref.setdefault(base.group(1), []).append(d.doc_id)
    out = project_path("data/queries")

    rows, qrels = [], []
    for need, sections, en, hi, hing in NEEDS:
        grades: dict[str, int] = {}
        for i, ref in enumerate(sections):
            g = 2 if i == 0 else 1
            for r in norm.equivalents(ref):        # the in-force (BNS) sections only
                for d in by_ref.get(r, []):
                    grades[d] = max(grades.get(d, 0), g)
        if not grades:
            print("no statute documents for", need, sections)
            continue
        for lang, text in (("en", en), ("hi", hi), ("hinglish", hing)):
            qid = f"E4-{need}-{lang}"
            rows.append({"id": qid, "text": text, "lang": lang, "state": "delhi", "incident_date": "2025-03-01",
                         "need_id": need, "sections": sections})
            qrels += [(qid, d, g) for d, g in sorted(grades.items())]
    _write(out, "e4_multilingual", rows, qrels, "team")

    rows, qrels = [], []
    for topic, text, sections in TOPICS:
        refs = set(sections) | {e for s in sections for e in norm.equivalents(s)}
        rel = sorted(d.doc_id for d in docs if d.doc_type.value == "precedent" and refs & set(d.statutes_cited))
        for state in STATES:
            qid = f"E6-{topic}-{state}"
            rows.append({"id": qid, "text": text, "lang": "en", "state": state, "topic": topic, "sections": sections})
            qrels += [(qid, d, 1) for d in rel]
        print(f"  E6 {topic}: {len(rel)} topically relevant precedents")
    _write(out, "e6_jurisdiction", rows, qrels, "citation")


def _write(out: Path, name: str, rows, qrels, judge: str) -> None:
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "qrels" / f"{name}.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["query_id", "doc_id", "grade", "judge"])
        w.writerows([q, d, g, judge] for q, d, g in qrels)
    print(f"{name}: {len(rows)} queries, {len(qrels)} judgments")


if __name__ == "__main__":
    main()
