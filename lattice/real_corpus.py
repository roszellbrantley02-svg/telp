"""
lattice/real_corpus.py - real Wikipedia extracts for unsupervised discovery test.

Pulled live from Wikipedia API on 2026-05-21. 12 entities across:
  Countries: Germany, France, Japan
  Animals:   Lion, Penguin, Shark
  People:    Einstein, Curie, Newton, Darwin
  Things:    Telephone, Internet
"""

WIKI_TEXTS = {
    "Germany": (
        "Germany, officially the Federal Republic of Germany, is a country in "
        "Western and Central Europe. It lies between the Baltic Sea and the "
        "North Sea to the north with the Alps to the south. Its sixteen "
        "constituent states have a total population of over 82 million, "
        "making it the most populous member state of the European Union (EU). "
        "Germany borders Denmark to the north; Poland and the Czech Republic "
        "to the east; Austria and Switzerland to the south; and France, "
        "Luxembourg, Belgium, and the Netherlands to the west. The nation's "
        "capital and most populous city is Berlin and its main financial "
        "centre is Frankfurt; the largest urban area is the Ruhr."
    ),
    "France": (
        "France, officially the French Republic, is a country primarily "
        "located in Western Europe. Its overseas regions and territories "
        "include French Guiana in South America, Saint Pierre and Miquelon "
        "in the North Atlantic, the French West Indies, and many islands in "
        "Oceania and the Indian Ocean. Metropolitan France shares borders "
        "with Belgium and Luxembourg to the north; Germany to the northeast; "
        "Switzerland to the east; Italy and Monaco to the southeast; "
        "Andorra and Spain to the south; and a maritime border with the "
        "United Kingdom to the northwest. Its capital, largest city and "
        "main cultural and economic centre is Paris."
    ),
    "Japan": (
        "Japan is an island country in East Asia. Located in the Pacific "
        "Ocean off the northeast coast of the Asian mainland, it is bordered "
        "to the west by the Sea of Japan and extends from the Sea of "
        "Okhotsk in the north to the East China Sea in the south. The "
        "Japanese archipelago consists of four major islands alongside "
        "14,121 smaller islands. Japan is divided into 47 administrative "
        "prefectures and eight traditional regions, and around 75% of its "
        "terrain is mountainous and heavily forested, concentrating its "
        "agriculture and highly urbanized population along its eastern "
        "coastal plains. Tokyo is the country's capital and largest city."
    ),
    "Lion": (
        "The lion is a large cat of the genus Panthera, currently ranging "
        "only in Sub-Saharan Africa and India. It has a muscular, "
        "broad-chested body; a short, rounded head; round ears; and a dark, "
        "hairy tuft at the tip of its tail. It is sexually dimorphic; adult "
        "male lions are larger than females and have a prominent mane that "
        "extends from the head to the shoulders and chest."
    ),
    "Penguin": (
        "Penguins are a group of flightless semi-aquatic sea birds which "
        "live almost exclusively in the Southern Hemisphere."
    ),
    "Shark": (
        "Sharks are a group of elasmobranch cartilaginous fishes "
        "characterized by a ribless endoskeleton, dermal denticles, five to "
        "seven gill slits on each side, and pectoral fins that are not fused "
        "to the head. Modern sharks are classified within the division "
        "Selachii and are the sister group to the Batomorphi."
    ),
    "Einstein": (
        "Albert Einstein was a German-born theoretical physicist best known "
        "for developing the theory of relativity. Einstein also made "
        "important contributions to quantum theory. His mass-energy "
        "equivalence formula E=mc2, which arises from special relativity, "
        "has been called the world's most famous equation. He received the "
        "1921 Nobel Prize in Physics for his services to theoretical "
        "physics, and especially for his discovery of the law of the "
        "photoelectric effect."
    ),
    "Curie": (
        "Marie Curie was a Polish and naturalised-French physicist and "
        "chemist. She shared the 1903 Nobel Prize in Physics with her "
        "husband Pierre Curie for their joint researches on the "
        "radioactivity phenomena discovered by Professor Henri Becquerel. "
        "She won the 1911 Nobel Prize in Chemistry for the discovery of "
        "the elements radium and polonium, by the isolation of radium and "
        "the study of the nature and compounds of this remarkable element."
    ),
    "Newton": (
        "Sir Isaac Newton was an English polymath who was a mathematician, "
        "physicist, astronomer, alchemist, theologian, author and inventor. "
        "He was a key figure in the Scientific Revolution and the "
        "Enlightenment that followed. His book Philosophiae Naturalis "
        "Principia Mathematica, first published in 1687, achieved the first "
        "great unification in physics and established classical mechanics. "
        "Newton also made seminal contributions to optics, and shares "
        "credit with the German mathematician Gottfried Wilhelm Leibniz "
        "for formulating infinitesimal calculus, although he developed "
        "calculus years before Leibniz."
    ),
    "Darwin": (
        "Charles Robert Darwin was an English naturalist, geologist, and "
        "biologist, widely known for his contributions to evolutionary "
        "biology. His proposition that all species of life have descended "
        "from a common ancestor is now generally accepted and considered a "
        "fundamental scientific concept. In a joint presentation with "
        "Alfred Russel Wallace, he introduced his scientific theory that "
        "this branching pattern of evolution resulted from a process he "
        "called natural selection."
    ),
    "Telephone": (
        "A telephone is a telecommunications device that enables two or "
        "more users to conduct a conversation when they are too far apart "
        "to be easily heard directly. A telephone converts sound, typically "
        "and most efficiently the human voice, into electronic signals that "
        "are transmitted via cables and other communication channels to "
        "another telephone which reproduces the sound to the receiving user."
    ),
    "Internet": (
        "The Internet is the global system of interconnected computer "
        "networks that uses the Internet protocol suite (TCP/IP) to "
        "communicate between networks and devices."
    ),
}


def split_sentences(text: str) -> list[str]:
    """Simple sentence splitter — splits on . ! ? followed by space."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def get_all_sentences() -> list[str]:
    """Return all sentences from all Wikipedia extracts."""
    all_sents = []
    for entity, text in WIKI_TEXTS.items():
        all_sents.extend(split_sentences(text))
    return all_sents


def get_sentences_with_source() -> list[tuple[str, str]]:
    """Return (entity_source, sentence) pairs."""
    out = []
    for entity, text in WIKI_TEXTS.items():
        for s in split_sentences(text):
            out.append((entity, s))
    return out


if __name__ == "__main__":
    sents = get_all_sentences()
    print(f"Total sentences from {len(WIKI_TEXTS)} Wikipedia summaries: {len(sents)}")
    print(f"\nSample sentences:")
    for s in sents[:10]:
        print(f"  - {s[:100]}")
