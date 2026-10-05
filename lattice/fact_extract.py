"""
lattice/fact_extract.py - read (subject, relation, object) facts out of
stored sentences, by rule.

Telp's memory is sentences. The fact layer (lattice/facts.py) wants
Facts - "Galileo Galilei --born_in--> Pisa" - each pointing back at the
sentence it was read from. This module is that bridge, and it is
deliberately simple: regular expressions and small word lists, no model
and no statistics. A pattern either fits the sentence or nothing is read
from it.

The hard part is not the verbs, it is WHO a sentence is about. The
Galileo article says "He taught at Padua" (Galileo), "Born in Pisa,
Galileo ..." (Galileo), but also "His father, Vincenzo Galilei, was a
lutenist" (NOT Galileo) and "The University of Padua was founded in 1222"
(the university). So every sentence first gets a subject:

  * He/She/It/They, or the possessives His/Her/Its/Their, mean the
    article subject - when they fit it. A "She" in an article that has
    been saying "he" is somebody else, and "It" is never a person.
  * A name means the article subject only when it IS that subject's name
    or short name ("Galileo", "Newton", "Curie").
  * Any other proper name is its own subject ("University of Padua",
    canonical name without "The").
  * Anything else - a common noun, a list of names, "My dog" - gives no
    facts at all.

Then the predicate is matched against relation patterns (the list is in
KNOWN_RELATION_PATTERNS). Objects end at the clause boundary and are at
most MAX_OBJECT_WORDS words; sentences that are questions, negated
("not", "never", "no longer"), hedged ("may have", "is thought to") or
in the first/second person are skipped whole. When in doubt the answer
is nothing: Telp's promise is that he never states what he cannot back
up, and every Fact here is backed by the exact sentence it cites.

extract_facts() reads one text. extract_from_rows() reads memory rows in
order and carries each article's context - its subject, the pronoun the
article itself uses for it, what kind of thing it is - from row to row.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

from lattice.facts import RELATIONS, Fact, strip_accents
from lattice.facts import norm_key as _norm_key_exact

__all__ = ["extract_facts", "extract_from_rows", "KNOWN_RELATION_PATTERNS",
           "MAX_OBJECT_WORDS"]

MAX_OBJECT_WORDS = 8


_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.I)
_SPACES = re.compile(r"\s+")


@lru_cache(maxsize=1 << 16)
def norm_key(s: str) -> str:
    """lattice.facts.norm_key, cached, with a fast path for plain ASCII
    (where accent stripping changes nothing). The extractor compares
    names thousands of times; this keeps 20,000 rows well under a
    second or two."""
    if not s.isascii():
        return _norm_key_exact(s)
    s = s.lower().strip().strip(" \t\n.,;:!?\"'()[]")
    return _SPACES.sub(" ", _ARTICLE.sub("", s))

# What each relation is read from (documentation, and a test checks that
# every relation the extractor emits is listed here).
KNOWN_RELATION_PATTERNS: dict[str, tuple[str, ...]] = {
    "born_in": ("X was born in P", "Born in P, X ...", "X, born in P, ..."),
    "born_on": ("X (15 February 1564 – 8 January 1642) was ...",
                "X (c. 23 April 1564 – ...)  [qualifier circa=yes]",
                "X was born on 15 February 1564", "X (born 15 March 1950)"),
    "born_year": ("X (1564–1642) was ...", "X (born 1950) is ...",
                  "X was born in 1564"),
    "died_in": ("X died in P",),
    "died_on": ("X (... – 8 January 1642) was ...", "X died on 8 January 1642"),
    "died_year": ("X (... – 1642) was ...", "X died in 1642"),
    "nationality": ("X was an Italian astronomer",
                    "X was a Polish and naturalised-French physicist",
                    "X was a Serbian-American inventor"),
    "occupation": ("X was an Italian astronomer, physicist and engineer",
                   "X worked as a patent clerk"),
    "instance_of": ("X is a city in R, C", "It is a gas giant",
                    "X is the fifth planet from the Sun"),
    "alias": ("X (dates), known as Y, was ...", "X is also known as Y",
              "X has been called Y"),
    "located_in": ("X is a city in R, C", "X is located in P",
                   "... born in Woolsthorpe, Lincolnshire"),
    "country": ("X is a city in Tuscany, Italy",
                "X is the largest city in Iceland"),
    "capital": ("Its capital (and largest city) is Y",
                "The capital of X is Y", "X's capital is Y",
                "Y is the capital of X"),
    "capital_of": ("Y is the capital of X", "Its capital is Y"),
    "official_language": ("The official language of X is Y",
                          "Its official languages are Y and Z"),
    "currency": ("Its currency is the euro", "The currency of X is Y"),
    "population": ("X has a population of about N (as of 2021)",
                   "Its population is N"),
    "area": ("X has an area of N square kilometres",),
    "educated_at": ("X studied (field) at U (and later at V)",
                    "X was educated at U", "X graduated from U",
                    "X attended U"),
    "taught_at": ("X taught (field) at U from 1592 to 1610",
                  "X was professor of ... at U"),
    "worked_at": ("X worked at/for O - only when O looks like an "
                  "organisation, never a person",),
    "lived_in": ("X lived in P", "X moved to P", "X settled in P"),
    "member_of": ("X was a member of O", "X was elected a Fellow of O",
                  "X joined O"),
    "spouse": ("X married Y (in 1895)", "X was married to Y",
               "His wife, Y, ..."),
    "parent": ("His father, Y, was ...", "X was the son of Y and Z"),
    "child": ("Her son, Y, ...", "(inverse of parent)"),
    "award": ("X received / won / was awarded the Nobel Prize in Physics "
              "in 1903 and ...",),
    "known_for": ("X is (best) known for Y",),
    "discovered": ("X discovered A and B in 1610", "Y was discovered by X"),
    "invented": ("X invented Y", "Y was invented by X"),
    "developed": ("X developed / formulated / devised Y",),
    "wrote": ("X wrote Hamlet, Macbeth and Romeo and Juliet",
              "X is the author of W", "W was written by X"),
    "composed": ("X composed W",),
    "painted": ("X painted W",),
    "founded": ("X founded O (in 1850)", "O was founded by X"),
    "founded_by": ("O was founded by X", "X founded O"),
    "founded_year": ("O was founded in 1222", "X founded O in 1850"),
    "author": ("W was written by X",),
    "published_year": ("W was published in 1605",),
    "headquarters": ("O is headquartered in P", "O is based in P"),
    "part_of": ("X is part of Y", "X is the largest planet in the Solar "
                "System"),
    "orbits": ("X orbits Y", "X is the fifth planet from the Sun",
               "X is a moon of Y"),
    "pronoun": ("He/His ... or She/Her ... in the subject's own article - "
                "never guessed",),
}


# ─── word lists ─────────────────────────────────────────────────────

_MONTH_NAMES = ("January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November",
                "December")
_MONTH_OF = {m.lower(): m for m in _MONTH_NAMES}
_MONTH_OF.update({m[:3].lower(): m for m in _MONTH_NAMES})
_MONTH_OF["sept"] = "September"
_MONTH = (r"(?:January|February|March|April|May|June|July|August|"
          r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|"
          r"Aug|Sept|Sep|Oct|Nov|Dec)\.?(?![a-z])")

# adjective -> country (None when there is no single modern country)
_DEMONYMS: dict[str, str | None] = {
    "American": "United States", "English": "England",
    "British": "United Kingdom", "Scottish": "Scotland", "Welsh": "Wales",
    "Irish": "Ireland", "French": "France", "German": "Germany",
    "Italian": "Italy", "Spanish": "Spain", "Portuguese": "Portugal",
    "Dutch": "Netherlands", "Belgian": "Belgium", "Swiss": "Switzerland",
    "Austrian": "Austria", "Polish": "Poland", "Czech": "Czech Republic",
    "Slovak": "Slovakia", "Hungarian": "Hungary", "Romanian": "Romania",
    "Bulgarian": "Bulgaria", "Serbian": "Serbia", "Croatian": "Croatia",
    "Slovenian": "Slovenia", "Slovene": "Slovenia", "Bosnian": None,
    "Montenegrin": "Montenegro", "Macedonian": "North Macedonia",
    "Albanian": "Albania", "Greek": "Greece", "Turkish": "Turkey",
    "Russian": "Russia", "Ukrainian": "Ukraine", "Belarusian": "Belarus",
    "Lithuanian": "Lithuania", "Latvian": "Latvia", "Estonian": "Estonia",
    "Finnish": "Finland", "Swedish": "Sweden", "Norwegian": "Norway",
    "Danish": "Denmark", "Icelandic": "Iceland", "Chinese": "China",
    "Japanese": "Japan", "Korean": None, "Indian": "India",
    "Pakistani": "Pakistan", "Bangladeshi": "Bangladesh",
    "Nepalese": "Nepal", "Afghan": "Afghanistan", "Iranian": "Iran",
    "Iraqi": "Iraq", "Syrian": "Syria", "Lebanese": "Lebanon",
    "Israeli": "Israel", "Jordanian": "Jordan", "Saudi": "Saudi Arabia",
    "Egyptian": "Egypt", "Moroccan": "Morocco", "Algerian": "Algeria",
    "Tunisian": "Tunisia", "Libyan": "Libya", "Nigerian": "Nigeria",
    "Ghanaian": "Ghana", "Kenyan": "Kenya", "Ethiopian": "Ethiopia",
    "Somali": "Somalia", "Ugandan": "Uganda", "Tanzanian": "Tanzania",
    "Zimbabwean": "Zimbabwe", "Mexican": "Mexico", "Canadian": "Canada",
    "Brazilian": "Brazil", "Argentine": "Argentina",
    "Argentinian": "Argentina", "Chilean": "Chile", "Peruvian": "Peru",
    "Colombian": "Colombia", "Venezuelan": "Venezuela", "Cuban": "Cuba",
    "Jamaican": "Jamaica", "Haitian": "Haiti", "Australian": "Australia",
    "Filipino": "Philippines", "Indonesian": "Indonesia",
    "Malaysian": "Malaysia", "Singaporean": "Singapore", "Thai": "Thailand",
    "Vietnamese": "Vietnam", "Burmese": "Myanmar", "Cambodian": "Cambodia",
    "Mongolian": "Mongolia", "Kazakh": "Kazakhstan", "Uzbek": "Uzbekistan",
    "Georgian": "Georgia", "Armenian": "Armenia",
    "Azerbaijani": "Azerbaijan", "Cypriot": "Cyprus", "Maltese": "Malta",
    "Luxembourgish": "Luxembourg",
    # historical / regional
    "Roman": None, "Prussian": None, "Bohemian": None, "Venetian": None,
    "Florentine": None, "Byzantine": None, "Ottoman": None, "Soviet": None,
    "Yugoslav": None, "Persian": None, "Flemish": None, "Bavarian": None,
    "Genoese": None, "Neapolitan": None, "Sicilian": None, "Catalan": None,
    "Austro-Hungarian": None, "Czechoslovak": None,
}

_COUNTRIES = frozenset(norm_key(c) for c in """
Afghanistan|Albania|Algeria|Andorra|Angola|Argentina|Armenia|Australia|
Austria|Azerbaijan|Bahamas|Bahrain|Bangladesh|Barbados|Belarus|Belgium|
Belize|Benin|Bhutan|Bolivia|Bosnia and Herzegovina|Botswana|Brazil|Brunei|
Bulgaria|Burkina Faso|Burundi|Cambodia|Cameroon|Canada|Chad|Chile|China|
Colombia|Comoros|Congo|Costa Rica|Croatia|Cuba|Cyprus|Czech Republic|
Czechia|Denmark|Djibouti|Dominica|Dominican Republic|Ecuador|Egypt|
El Salvador|England|Eritrea|Estonia|Eswatini|Ethiopia|Fiji|Finland|France|
Gabon|Gambia|Georgia|Germany|Ghana|Greece|Grenada|Guatemala|Guinea|Guyana|
Haiti|Honduras|Hungary|Iceland|India|Indonesia|Iran|Iraq|Ireland|Israel|
Italy|Jamaica|Japan|Jordan|Kazakhstan|Kenya|Kosovo|Kuwait|Kyrgyzstan|Laos|
Latvia|Lebanon|Lesotho|Liberia|Libya|Liechtenstein|Lithuania|Luxembourg|
Madagascar|Malawi|Malaysia|Maldives|Mali|Malta|Mauritania|Mauritius|Mexico|
Moldova|Monaco|Mongolia|Montenegro|Morocco|Mozambique|Myanmar|Namibia|Nepal|
Netherlands|New Zealand|Nicaragua|Niger|Nigeria|North Korea|North Macedonia|
Northern Ireland|Norway|Oman|Pakistan|Panama|Papua New Guinea|Paraguay|Peru|
Philippines|Poland|Portugal|Qatar|Romania|Russia|Rwanda|Samoa|San Marino|
Saudi Arabia|Scotland|Senegal|Serbia|Seychelles|Sierra Leone|Singapore|
Slovakia|Slovenia|Somalia|South Africa|South Korea|South Sudan|Spain|
Sri Lanka|Sudan|Suriname|Sweden|Switzerland|Syria|Taiwan|Tajikistan|
Tanzania|Thailand|Togo|Tonga|Trinidad and Tobago|Tunisia|Turkey|
Turkmenistan|Uganda|Ukraine|United Arab Emirates|United Kingdom|
United States|United States of America|Uruguay|Uzbekistan|Vanuatu|
Vatican City|Venezuela|Vietnam|Wales|Yemen|Zambia|Zimbabwe
""".replace("\n", "").split("|"))

# category nouns that make the subject a place
_PLACE_CLASSES = frozenset("""
city town village hamlet municipality commune comune country nation state
province region county district island islands archipelago peninsula river
lake mountain volcano sea ocean continent metropolis port settlement
borough suburb territory kingdom republic principality duchy emirate
city-state valley desert bay strait canal capital oblast prefecture canton
""".split())

# nouns that make the subject a person (plus the suffix rule below)
_OCCUPATIONS = frozenset("""
astronomer physicist engineer mathematician chemist biologist philosopher
writer poet playwright novelist author painter sculptor architect composer
musician singer songwriter actor actress director producer politician
statesman diplomat lawyer judge physician doctor surgeon nurse inventor
entrepreneur businessman businesswoman economist historian scientist
theologian priest monk nun bishop pope king queen emperor empress prince
princess general soldier admiral explorer navigator journalist editor
publisher teacher professor scholar linguist translator critic essayist
lutenist theorist polymath naturalist botanist zoologist geologist
geographer cartographer astronaut pilot aviator athlete footballer
cricketer boxer comedian dancer choreographer photographer designer
illustrator cartoonist programmer statistician psychologist psychiatrist
sociologist anthropologist archaeologist activist revolutionary
philanthropist industrialist banker merchant farmer chef conductor pianist
violinist guitarist drummer rapper model presenter broadcaster minister
president chancellor senator governor mayor monarch ruler leader saint
martyr prophet emperor pharaoh sultan tsar khan warrior laureate
researcher inventor educator lecturer chronicler courtier knight lord
lady duke duchess count countess baron noblewoman nobleman officer
commander captain spy detective executive investor manager founder
""".split())
_OCC_SUFFIXES = ("ist", "ologist", "ician", "grapher", "smith", "wright",
                 "eer", "man", "woman")
_NOT_OCCUPATIONS = frozenset("""
man woman human list mist fist wrist twist exist gist cyst midst career
veneer persist assist resist insist consist desist amethyst german roman
ottoman talisman
""".split())

_AWARD_WORDS = frozenset("""
Prize Award Awards Medal Cross Order Cup Trophy Oscar Oscars Grammy Emmy
Tony Fellowship Pulitzer Nobel Globe BAFTA Booker Legion Honour Honor
Laureate Knighthood Decoration Garter
""".split())

_ORG_WORDS = frozenset("""
University College School Academy Institute Institution Laboratory
Laboratories Labs Lab Observatory Company Corporation Corp Inc Ltd LLC
Group Society Association Foundation Council Office Agency Bureau Ministry
Department Museum Library Hospital Bank Church Party Army Navy Court
Parliament Congress Senate Committee Commission Organization Organisation
Union League Federation Club Team Orchestra Theatre Theater Studio Studios
Press Records Motors Electric Works Systems Technologies Gymnasium
Seminary Conservatory Polytechnic Center Centre Service Board Trust Fund
Authority Administration Station Enterprises Industries Airlines Railway
""".split())
# a "Trinity College, Cambridge" name keeps its place after the comma
_INSTITUTION_WORDS = frozenset("""
College Hall School Academy Institute Gymnasium Seminary Lyceum
Conservatory Polytechnic University
""".split())

# lower-case words allowed inside proper names ("Weil der Stadt")
_CONNECTORS = frozenset("""
of de de' d' di da del della delle dei degli der den des du van von la le
las los upon am sur en y bin ibn al el zu ser dos das do
""".split())
_HONORIFICS = frozenset({"sir", "dame"})
_NAME_SUFFIXES = frozenset({"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"})
# first words that make a title a place or an organisation, not a person
_NOT_PERSON_TITLE = frozenset("""
New San Santa Saint St. Los Las El La Le Mount Lake Port Fort Cape North
South East West Upper Lower Great Little Old Royal United Republic Kingdom
The
""".split())

# capitalised sentence-openers that are never a subject's name
_NOT_SUBJECT = frozenset("""
In On At After Before During Later Also However Today Here There This That
These Those Many Most Some Several Both Each All Such Other Another A An
His Her Its Their He She It They We You I My Our Your What Which Who Whom
Whose When Where Why How If Although Though As For From With By Of To Like
Unlike Despite Following Now Since While Because Once Then Thus Hence
Born Known According Between Among Under Over Within Without Upon Around
Along Throughout Toward Towards Beyond Near Besides Meanwhile Moreover
Furthermore Additionally Notably Eventually Subsequently Initially
Originally Currently Recently Nevertheless Nonetheless Instead Indeed Yet
Still Even Only Just Perhaps Unfortunately Ultimately Finally First Second
Third Last Lastly Together Overall Generally Usually Often Sometimes
Similarly Likewise Conversely Nobody None No Not Every Any Neither Either
Much More Less Few Several
""".split())

_ABBREV = frozenset("""
st mt ft jr sr dr mr mrs ms prof gen col lt capt sgt rev fr no vs etc inc
ltd co bros corp c ca approx est e.g i.e u.s u.k jan feb mar apr jun jul
aug sep sept oct nov dec
""".split())

_ORDINALS = frozenset("""
first second third fourth fifth sixth seventh eighth ninth tenth eleventh
twelfth
""".split())
_DIRECTIONS = frozenset("""
northern southern eastern western central north-western northwestern
north-eastern northeastern south-western southwestern south-eastern
southeastern north south east west
""".split())

# a class noun phrase ends at any of these
_NP_STOP = frozenset("""
who which that whose whom where when while in on at of for from with by to
as known born based located situated considered regarded described best
widely often noted famous renowned during after before between under over
since until like than or but also one is was has had are were became the a
an his her its their this these those such more most many several some
first only other very both all each there it he she they near across along
around about against among through throughout via within without including
especially notably primarily mainly nicknamed called named sometimes
usually generally later then active working living whose
""".split())
# nouns that mean nothing without their "of ..." ("a student of Galileo",
# "a co-winner of the prize", "a kind of fish")
_RELATIONAL = frozenset("""
student pupil friend son daughter wife husband member part one father
mother brother sister child descendant ancestor disciple follower rival
colleague partner servant citizen resident native relative cousin nephew
niece grandson granddaughter uncle aunt contemporary successor predecessor
kind type sort form group number series set piece variety collection pair
couple lot total portion fraction majority minority branch subset version
example instance unit co-winner winner recipient holder owner victim
co-founder founder head leader
""".split())
_LY_NOUNS = frozenset("""
family lily rally ally assembly monopoly butterfly fly jelly belly bully
supply reply anomaly homily folly italy
""".split())
_CLASS_WORD = re.compile(r"^[a-z][a-z'-]*$")
# "-ed" words that are nouns, and what follows a participle phrase
# ("a rock band formed in Liverpool")
_ED_NOUNS = frozenset("bed red seed breed shed steed creed reed sled deed "
                      "speed weed feed greed need".split())
_PARTICIPLE_NEXT = frozenset("in by at on from after for to with as during "
                             "under into".split())


# ─── sentence filters ───────────────────────────────────────────────
#
# A sentence containing any of these words is skipped whole. They are
# looked up in the set of the sentence's words (one findall) rather than
# with one big regex: a word alternation tried at every character costs
# ~40 µs a sentence, the set lookup a few.

_NEGATION_WORDS = frozenset("not never no nor neither none nobody nothing "
                            "nowhere cannot".split())
# case-sensitive: "May" is a month, "Will" a name
_HEDGE_WORDS = frozenset("""
may might could would should will possibly perhaps probably presumably
allegedly reportedly supposedly purportedly apparently arguably rumored
rumoured speculated alleged disputed uncertain unclear unconfirmed
unverified likely unlikely if unless whether claimed
""".split())
_HEDGE_OPENERS = frozenset("If Unless Perhaps Possibly Probably Allegedly "
                           "Reportedly Supposedly Maybe According Legend"
                           .split())
# "believed to", "thought to", "said to", "according to", "legend has it"
_HEDGE_PAIRS = frozenset({("believed", "to"), ("thought", "to"),
                          ("said", "to"), ("according", "to"),
                          ("legend", "has")})
# first and second person: about the user or the conversation, not the
# world
_HEDGE_PAIR_FIRST = frozenset({"believed", "thought", "said", "according",
                               "According", "legend", "Legend"})
_PERSONAL_WORDS = frozenset("me my mine myself we us our ours ourselves "
                            "you your yours yourself".split())
_PERSONAL_OPENERS = frozenset("My We Our You Your Me".split())
_SENT_WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
# a capital "I" is the speaker only after a lower-case word or at the
# start ("Elizabeth I was" and "World War I" are names)
_FIRST_PERSON_I = re.compile(r"(?:^|(?<![A-Za-z])[a-z]+,?\s+)I\b(?!\.)")


def _skip_sentence(s: str) -> bool:
    """Questions, negations, hedges and first/second-person sentences
    state nothing Telp may repeat as a fact."""
    if "?" in s:
        return True
    words = _SENT_WORD.findall(s)
    if not words:
        return True
    ws = set(words)
    if ws & _HEDGE_WORDS or ws & _PERSONAL_WORDS or ws & _NEGATION_WORDS:
        return True
    first = words[0]
    if first in _HEDGE_OPENERS or first in _PERSONAL_OPENERS \
            or first.lower() in _NEGATION_WORDS or "n't" in s:
        return True
    if ws & _HEDGE_PAIR_FIRST:
        for a, b in zip(words, words[1:]):
            if (a.lower(), b) in _HEDGE_PAIRS:
                return True
    return "I" in ws and bool(_FIRST_PERSON_I.search(s))


_EXPLETIVE_IT = re.compile(
    r"^It\s+(?:is|was|has\s+been|had\s+been|seems|seemed|appears|appeared|"
    r"remains)\s+(?:\w+\s+){0,3}?(?:that|to|whether|how|why|when)\b")

_CITE = re.compile(r"\[(?:\d+|[a-z ]+needed|note \d+)\]")
_WS = re.compile(r"\s+")


# ─── dates ──────────────────────────────────────────────────────────

_ERA = r"(?:\s*(?P<era>BC|BCE|AD|CE)\b)?"
_DATE_FORMS = (
    ("dmy", re.compile(rf"^(?P<d>\d{{1,2}})\s+(?P<m>{_MONTH})\s+"
                       rf"(?P<y>\d{{1,4}}){_ERA}$")),
    ("mdy", re.compile(rf"^(?P<m>{_MONTH})\s+(?P<d>\d{{1,2}}),?\s+"
                       rf"(?P<y>\d{{1,4}}){_ERA}$")),
    ("my", re.compile(rf"^(?P<m>{_MONTH})\s+(?P<y>\d{{1,4}}){_ERA}$")),
    ("y", re.compile(rf"^(?:(?P<pre>AD|CE)\s+)?(?P<y>\d{{1,4}}){_ERA}$")),
)
_CIRCA = re.compile(r"^(?:c\.|ca\.|circa|approximately|about|around)\s*",
                    re.I)
# a date written inside a sentence (full date or month + year)
_DMY = rf"\d{{1,2}}\s+{_MONTH}\s+\d{{3,4}}"
_MDY = rf"{_MONTH}\s+\d{{1,2}},?\s+\d{{3,4}}"
_YEAR = r"\d{3,4}(?:\s*(?:BC|BCE)\b)?(?![\d])(?!,\d)"


@dataclass(frozen=True)
class _When:
    date: str | None      # "15 February 1564" (always day-month-year)
    year: str             # "1564", or "470 BC"
    circa: bool
    number: int           # signed year for sanity checks (BC < 0)


def _parse_date(text: str) -> _When | None:
    """One date expression - "15 February 1564", "February 15, 1564",
    "c. 1564", "April 1564", "399 BC" - or None."""
    t = text.strip().rstrip(".,")
    circa = False
    m = _CIRCA.match(t)
    if m:
        circa, t = True, t[m.end():]
    for form, rx in _DATE_FORMS:
        m = rx.match(t)
        if not m:
            continue
        y = int(m.group("y"))
        era = (m.group("era") or "").upper()
        explicit_ad = bool(era in ("AD", "CE")
                           or (form == "y" and m.group("pre")))
        bc = era in ("BC", "BCE")
        if not (bc or explicit_ad) and not 100 <= y <= 2100:
            return None                   # "12" alone is not a year
        if y == 0 or y > 2100:
            return None
        year = f"{y} BC" if bc else str(y)
        date = None
        if form in ("dmy", "mdy"):
            d = int(m.group("d"))
            if not 1 <= d <= 31:
                return None
            month = _MONTH_OF[m.group("m").rstrip(".").lower()]
            date = f"{d} {month} {year}"
        return _When(date, year, circa, -y if bc else y)
    return None


def _life_dates(paren: str) -> list[tuple[str, str, tuple]]:
    """Birth/death facts from a lead parenthesis: "15 February 1564 – 8
    January 1642", "c. 23 April 1564 – 23 April 1616", "1564–1642",
    "born 1950", "c. 470 – 399 BC". Other parts of the parenthesis
    (pronunciation, birth name) are separated by ";" and ignored."""
    for seg in re.split(r"\s*;\s*", paren):
        seg = seg.strip().strip(",").strip()
        if not seg:
            continue
        m = re.match(r"^(?P<w>born|b\.|died|d\.)\s+(?P<d>.+)$", seg)
        if m:
            when = _parse_date(m.group("d"))
            if when:
                return _date_facts("born" if m.group("w")[0] == "b"
                                   else "died", when)
            continue
        parts = re.split(r"\s*[–—−]\s*|\s+-\s+|(?<=\d)-(?=\d|c\.)", seg)
        if len(parts) != 2:
            continue
        a, b = _parse_date(parts[0]), _parse_date(parts[1])
        if a and b and b.year.endswith(" BC") and not a.year.endswith(" BC") \
                and not re.search(r"\b(?:AD|CE)\b", parts[0]):
            # "c. 470 – 399 BC": the era written once covers both ends
            a = _When(a.date and a.date + " BC", a.year + " BC", a.circa,
                      -abs(a.number))
        if a and b and not 0 <= b.number - a.number <= 125:
            return []                      # not a lifespan
        facts: list[tuple[str, str, tuple]] = []
        if a:
            facts += _date_facts("born", a)
        if b:
            facts += _date_facts("died", b)
        if facts:
            return facts
    return []


def _date_facts(event: str, when: _When) -> list[tuple[str, str, tuple]]:
    quals = (("circa", "yes"),) if when.circa else ()
    out = []
    if when.date:
        out.append((f"{event}_on", when.date, quals))
    out.append((f"{event}_year", when.year, quals))
    return out


# ─── names ──────────────────────────────────────────────────────────

_TRAIL = ",;:!?)\"'."


def _is_name_word(w: str) -> bool:
    """A word that can be part of a proper name: capitalised ("Pisa",
    "ETH", "Skłodowska-Curie") or an elided particle ("d'Alembert")."""
    if not w:
        return False
    if w[0].isupper():
        return True
    return len(w) > 2 and w[1] == "'" and w[2].isupper()


def _scan_name(text: str, pos: int = 0, allow_the: bool = False,
               max_words: int = 8) -> tuple[str, int, bool]:
    """The proper name that starts at text[pos:]: capitalised words joined
    by name particles ("University of Padua", "Weil der Stadt", "J. R. R.
    Tolkien"). Stops at a comma, a lower-case word or the sentence end.
    Returns (name, end index, possessive?) - name "" when there is none."""
    words: list[str] = []
    pending: list[str] = []
    end = start = pos
    while start < len(text) and text[start] == " ":
        start += 1
    # texts are tidied to single spaces, so one split finds every token
    # and its offset (a regex call per token was the extractor's hot spot)
    first = True
    for raw in text[start:].split(" ", 3 * max_words):
        if len(words) >= max_words or not raw or " " in raw:
            break
        tok_start, start = start, start + len(raw) + 1
        if first and allow_the and raw in ("the", "The"):
            pending.append(raw)
            first = False
            continue
        first = False
        if raw in ("de'", "d'"):
            core, trail = raw, ""
        else:
            core = raw.rstrip(_TRAIL)
            trail = raw[len(core):]
            # keep the dot of "St." and of initials ("J.")
            if trail.startswith(".") and core and (
                    core.lower() in _ABBREV
                    or (len(core) == 1 and core.isupper())):
                core, trail = core + ".", trail[1:]
        if not core:
            break
        poss = core[:-2] if core.endswith("'s") else None
        word = poss if poss is not None else core
        if _is_name_word(word) or (words and word.isdigit()
                                   and len(word) <= 2):
            words.extend(pending)
            pending = []
            words.append(word)
            end = tok_start + len(word)
            if poss is not None:
                return " ".join(words), end, True
            if trail:
                break
            continue
        if words and not trail and poss is None and (
                word in _CONNECTORS
                or (word == "the" and (pending or words)[-1] == "of")):
            pending.append(word)
            continue
        break
    if not words:
        return "", pos, False
    return " ".join(words), end, False


# a name object must end where the clause ends or a new detail begins;
# "the Warmian | chapter" was cut short by a lower-case word
_NAME_END = re.compile(
    r"\s*(?:$|[.;:,!)\"]|\s(?:and|or|but|in|on|at|to|from|where|which|who|"
    r"whom|whose|when|after|before|until|as|since|for|with|during|under|by|"
    r"near|between|while|then|aged|until|following|alongside|together|"
    r"along|via|through|was|is|were|are|had|has|having|being|becoming|"
    r"making|where)\b)")


def _scan_obj(text: str, pos: int = 0, allow_the: bool = True
              ) -> tuple[str, int]:
    """A name used as an object ("in Weil der Stadt", "of the Royal
    Society"); "" when there is none or it was cut short."""
    name, end, poss = _scan_name(text, pos, allow_the=allow_the)
    if not name or poss or not _NAME_END.match(text, end):
        return "", pos
    return name, end


_THE = re.compile(r"^(?:the|The)\s+")
_SIR = re.compile(r"^(?:Sir|Dame)\s+(?=\S+\s+\S)")
_COMMA_SP = re.compile(r",\s+")
_AND_NAME = re.compile(r"\s*,?\s+and\s+(?=[A-Z])")
_DISAMBIG = re.compile(r"\s*\([^)]*\)$")


@lru_cache(maxsize=1 << 14)
def _canon(name: str) -> str:
    """Canonical entity name: no leading "The", no "Sir"/"Dame"."""
    n = name.strip().rstrip(",;:")
    return _SIR.sub("", _THE.sub("", n))


def _name_tokens(name: str) -> set[str]:
    s = strip_accents(name).casefold()
    return {t for t in re.split(r"[\s\-,.']+", s)
            if t and t not in _CONNECTORS and t not in _HONORIFICS
            and t not in _NAME_SUFFIXES}


def _looks_like_full_name(name: str) -> bool:
    words = name.replace(",", " ").split()
    if not 1 <= len(words) <= 10 or not _is_name_word(words[0]):
        return False
    return all(_is_name_word(w) or w in _CONNECTORS
               or w in ("of", "the") for w in words)


def _looks_like_title(name: str) -> bool:
    """Could this be an article title (the "Title:" growth anchor)?"""
    base = _DISAMBIG.sub("", name).strip()
    words = base.split()
    if not 1 <= len(words) <= 8 or not _is_name_word(words[0]):
        return False
    return all(_is_name_word(w) or w in _CONNECTORS or w in ("of", "the",
                                                             "and")
               for w in words)


def _person_like_title(title: str) -> bool:
    """"Galileo Galilei" yes; "New York City", "University of Padua" no."""
    words = _DISAMBIG.sub("", title).split()
    if not 2 <= len(words) <= 4 or words[0] in _NOT_PERSON_TITLE:
        return False
    for w in words:
        if w in _CONNECTORS:
            continue
        if not _is_name_word(w) or w in _ORG_WORDS or w in _AWARD_WORDS \
                or norm_key(w) in _COUNTRIES or w.lower() in _PLACE_CLASSES:
            return False
    return True


def _is_country(name: str) -> bool:
    return norm_key(name) in _COUNTRIES


def _nationality(word: str) -> tuple[str, tuple] | None:
    """"Italian" -> ("Italian", ()); "Serbian-American" stays whole;
    "naturalised-French" -> ("French", (("naturalised", "yes"),))."""
    w = word.strip(",")
    quals: tuple = ()
    low = w.lower()
    if low.startswith(("naturalised-", "naturalized-")):
        w, quals = w.split("-", 1)[1], (("naturalised", "yes"),)
    if not w or not w[0].isupper():
        return None
    if w in _DEMONYMS:
        return w, quals
    parts = w.split("-")
    if len(parts) > 1 and all(p in _DEMONYMS for p in parts):
        return w, quals
    return None


def _is_occupation(item) -> bool:
    head = item[-1]
    if head in _NOT_OCCUPATIONS:
        return False
    return (head in _OCCUPATIONS or " ".join(item) in _OCCUPATIONS
            or head.endswith(_OCC_SUFFIXES))


# ─── context and output ─────────────────────────────────────────────

@dataclass
class _Context:
    """What the reader knows about the article it is in."""
    subject: str | None = None
    names: set[str] = field(default_factory=set)    # norm keys
    short: set[str] = field(default_factory=set)    # "galileo", "curie"
    kind: str | None = None          # "person" / "place" / "thing"
    pronoun: str | None = None       # "he" / "she", once the text says so
    plural: bool = False             # "The Beatles were ..." -> "They"
    pending: str | None = None       # pronoun this sentence used
    pronoun_said: bool = False       # pronoun fact already emitted
    classes: set[str] = field(default_factory=set)  # "city" -> "The city"

    @classmethod
    def about(cls, subject: str | None) -> "_Context":
        ctx = cls()
        if subject:
            ctx.adopt(subject)
        return ctx

    def adopt(self, subject: str) -> None:
        subject = subject.strip()
        self.subject = subject
        base = _DISAMBIG.sub("", subject).strip() or subject
        self.names |= {norm_key(subject), norm_key(base)}
        if _person_like_title(base):
            words = [w for w in base.split()
                     if w not in _CONNECTORS
                     and w.lower().rstrip(".") not in _NAME_SUFFIXES]
            self.short |= {norm_key(words[0]), norm_key(words[-1])}


@dataclass
class _Clause:
    subject: str
    is_ctx: bool                  # is this the article subject?
    pred: str                     # "was born in Pisa."
    kind: str | None = None       # kind hint for non-article subjects


class _Out:
    """Collects Facts with the provenance of the row being read."""

    def __init__(self) -> None:
        self.facts: list[Fact] = []
        self._seen: set = set()
        self.prov: tuple[str, str, int | None, str | None] = ("", "", None,
                                                               None)

    def add(self, subject: str, relation: str, obj: str,
            quals: Iterable[tuple[str, str]] = (),
            canonical: bool = False) -> None:
        """Record one fact. Subjects are canonicalised ("The University
        of Padua" -> "University of Padua") unless the caller says the
        name is canonical already - an article's own title stays as it
        is ("The Beatles")."""
        if not canonical:
            subject = _canon(subject)
        obj = _clean_obj(obj)
        if not subject or not obj or relation not in RELATIONS:
            return
        ko, ks = norm_key(obj), norm_key(subject)
        if not ko or ko == ks or obj.count(" ") >= MAX_OBJECT_WORDS:
            return                   # empty, self-referential or too long
        q = tuple(sorted({k: v for k, v in quals if v}.items())) \
            if quals else ()
        text, source, mid, created = self.prov
        key = (ks, relation, ko, q, mid, text)
        if key in self._seen:
            return
        self._seen.add(key)
        self.facts.append(Fact(subject, relation, obj, source=source,
                               text=text, memory_id=mid, created_at=created,
                               qualifiers=q))


_ABBREV_END = re.compile(r"(?:\b[A-Z]|\b(?:St|Jr|Sr|Inc|Ltd|Co|Bros))\.$")


@lru_cache(maxsize=1 << 14)
def _clean_obj(obj: str) -> str:
    o = _WS.sub(" ", obj).strip().strip("\"'").strip()
    while o and o[-1] in ",;:":
        o = o[:-1].rstrip()
    if o.endswith(".") and not _ABBREV_END.search(o):
        o = o[:-1].rstrip()
    return o


# which kind of subject each relation needs; a fact that contradicts the
# article's kind ("born in" for a city) means the subject was misread
_PERSON_ONLY = frozenset({
    "born_in", "born_on", "born_year", "died_in", "died_on", "died_year",
    "occupation", "nationality", "spouse", "parent", "child",
    "educated_at", "taught_at", "worked_at", "lived_in", "pronoun"})
_PLACE_ONLY = frozenset({"capital", "capital_of", "official_language",
                         "population", "area", "currency"})


def _emit(c: _Clause, ctx: _Context, out: _Out, rel: str, obj: str,
          quals: Iterable[tuple[str, str]] = ()) -> None:
    kind = c.kind or (ctx.kind if c.is_ctx else None)
    if rel in _PERSON_ONLY and kind in ("place", "thing"):
        return
    if rel in _PLACE_ONLY and kind == "person":
        return
    if c.is_ctx and ctx.kind is None:
        if rel in _PERSON_ONLY:
            ctx.kind = "person"
        elif rel in _PLACE_ONLY:
            ctx.kind = "place"
    before = len(out.facts)
    out.add(c.subject, rel, obj, quals, canonical=True)
    if c.is_ctx and ctx.pending and len(out.facts) > before:
        out.add(c.subject, "pronoun", ctx.pending, canonical=True)
        ctx.pending, ctx.pronoun_said = None, True


# ─── object spans ───────────────────────────────────────────────────

# where an object phrase ends: punctuation, a subordinate clause, a dash
_HARD_END = re.compile(
    r"\s*[.;:!?](?=\s|$)|\s*[–—]\s|\s+-\s"
    r"|,\s+(?=(?:which|who|whom|whose|where|when|while|whilst|although|"
    r"though|because|but|so|after|before|until|making|becoming|thus|"
    r"thereby|including|especially|leading|resulting|alongside|along|"
    r"together|then|later|following|despite|since|as|with|for|from|"
    r"in\s+which|and\s+(?:then|later|was|is|has|had|became|also))\b)"
    r"|\s+(?=(?:but|where|which|who|whom|whose|that|when|while|whilst|"
    r"although|though|because|whereas|thereby|thus|until|despite|"
    r"alongside|since|after|before|during|so\s+that|as\s+well\s+as)\b)")
# for discoveries and inventions: where/when/how it happened is not part
# of the thing ("the four largest moons of Jupiter | in 1610")
_THING_END = re.compile(
    r"\s+(?=(?:in|at|with|while|using|by|from|on|into|through|under|among|"
    r"during|for|near|whilst|as)\s)")
_ITEM_YEAR = re.compile(
    rf"\s+(?:in|on)\s+(?:(?:\d{{1,2}}\s+)?{_MONTH}\s+)?(?P<y>\d{{3,4}})"
    rf"(?:\s*BC)?(?=\s|$)")
_VAGUE_START = frozenset("""
that how what why whether it him her them this these those there himself
herself itself themselves many several some various numerous much more
most other others a an one two three new
""".split())
_DANGLING_END = frozenset("of the a an and or in on at to for with by from"
                          .split())
_TITLE_SMALL = frozenset("""
of the and a an in on to for with at by from or de la le du del von van
is are it you upon into over under nor but
""".split())
_WORK_KIND = re.compile(
    r"^(?:the|his|her|their)\s+(?:[a-z]+\s+)?(?:novels?|plays?|poems?|"
    r"books?|operas?|symphon(?:y|ies)|paintings?|songs?|albums?|essays?|"
    r"treatises?|traged(?:y|ies)|comed(?:y|ies)|musicals?|films?|series|"
    r"stor(?:y|ies)|novellas?|sonnets?|concertos?|oratorios?|requiem)\s+"
    r"(?=[A-Z0-9\"])")
# "and" titles that are one work, not two
_AND_TITLES = frozenset({
    "romeo and juliet", "pride and prejudice", "war and peace",
    "crime and punishment", "sense and sensibility", "antony and cleopatra",
    "troilus and cressida", "fathers and sons", "the prince and the pauper",
    "of mice and men", "venus and adonis"})


def _cut(text: str, rx: re.Pattern) -> str:
    m = rx.search(text)
    return text[:m.start()] if m else text


def _split_list(span: str, works: bool = False) -> list[str]:
    """"A, B and C" -> [A, B, C]. Only the final "and" of a comma list
    joins items, so "Hamlet, Macbeth and Romeo and Juliet" keeps "Romeo
    and Juliet" whole. A comma without a final "and" is an aside, not a
    list: only the first part is kept."""
    span = span.strip()
    if ", " in span:
        segs = [s.strip() for s in span.split(", ")]
        last = segs[-1]
        if last.startswith("and "):
            segs[-1] = last[4:]
        elif " and " in last:
            a, b = last.split(" and ", 1)
            segs[-1:] = [a, b]
        else:
            return [segs[0]]
        return segs
    if " and " not in span or re.search(r"\bbetween\b", span):
        return [span]
    if works:
        if norm_key(span) in _AND_TITLES:
            return [span]
        a, b = span.split(" and ", 1)
        if len(a.split()) == 1 and len(b.split()) == 1:
            return []        # "Hamlet and Macbeth" or one title? unsure.
        return [a, b]
    return [p for p in re.split(r"\s+and\s+", span) if p]


def _item_year(item: str) -> tuple[str, str | None]:
    m = _ITEM_YEAR.search(item)
    if not m:
        return item, None
    return item[:m.start()], m.group("y")


def _thing_ok(v: str, loose: bool = False) -> bool:
    """A discovery/invention/"known for" value: a specific noun phrase,
    not a pronoun or a vague quantity ("many comets"). loose allows "a
    ..." and "her ..." (known for "a series of ...", "her work on ...")."""
    words = v.split()
    if not 1 <= len(words) <= MAX_OBJECT_WORDS:
        return False
    first = words[0].lower()
    if first in _VAGUE_START and not (
            loose and len(words) > 1 and first in ("a", "an", "her")):
        return False
    if words[-1].lower() in _DANGLING_END:
        return False
    return any(ch.isalpha() for ch in v)


def _title_ok(v: str) -> bool:
    """Title-shaped: "Hamlet", "Romeo and Juliet", "the Mona Lisa"."""
    words = v.strip("\"").split()
    if len(words) > 1 and words[0] == "the":
        words = words[1:]
    if not 1 <= len(words) <= MAX_OBJECT_WORDS:
        return False
    if not (words[0][0].isupper() or words[0][0].isdigit()):
        return False
    if not (words[-1][0].isupper() or words[-1][0].isdigit()):
        return False
    return all(w[0].isupper() or w[0].isdigit() or w.lower() in _TITLE_SMALL
               for w in words)


def _award_ok(v: str) -> bool:
    core = re.sub(r"^(?:the|a|an)\s+", "", v)
    return _title_ok(core) and any(w.strip(",") in _AWARD_WORDS
                                   for w in core.split())


def _period(text: str) -> tuple[list[tuple[str, str]], int]:
    """Qualifiers right after an object: "from 1592 to 1610", "between
    1592 and 1610", "until 1610", "since 2001", "in 1610", "as Lucasian
    Professor of Mathematics" (role). Returns (qualifiers, chars used)."""
    quals: list[tuple[str, str]] = []
    pos = 0
    for _ in range(2):
        rest = text[pos:]
        m = _PERIOD.match(rest)
        if m:
            g = m.groupdict()
            if g["a"]:
                quals += [("from", g["a"]), ("to", g["b"])]
            elif g["c"]:
                quals += [("from", g["c"]), ("to", g["d"])]
            elif g["e"]:
                quals.append(("to", g["e"]))
            elif g["f"] or g["g"]:
                quals.append(("from", g["f"] or g["g"]))
            elif g["h"]:
                quals.append(("year", g["h"]))
            pos += m.end()
            continue
        m = _ROLE.match(rest)
        if m:
            role = re.sub(r"\s+(?:of|for|in|and)$", "", m.group("role"))
            if len(role.split()) <= 6:
                quals.append(("role", role))
            pos += m.end()
            continue
        break
    return quals, pos


_PERIOD = re.compile(
    r"\s*,?\s*(?:from\s+(?P<a>\d{4})\s+(?:to|until|till|through)\s+"
    r"(?P<b>\d{4})|between\s+(?P<c>\d{4})\s+and\s+(?P<d>\d{4})"
    r"|until\s+(?P<e>\d{4})|since\s+(?P<f>\d{4})|from\s+(?P<g>\d{4})"
    r"|in\s+(?P<h>\d{4}))(?![\d])")
_ROLE = re.compile(
    r"\s+as\s+(?:an?\s+|the\s+|its\s+)?(?P<role>[A-Z][\w'-]*"
    r"(?:\s+(?:of|for|in|and|[A-Z][\w'-]*))*)")


def _scan_org(text: str, pos: int = 0) -> tuple[str, int]:
    """An organisation name, keeping a place after the comma for
    colleges: "Trinity College, Cambridge"."""
    name, end = _scan_obj(text, pos)
    if not name:
        return "", pos
    if name.split()[-1] in _INSTITUTION_WORDS:
        m = _COMMA_SP.match(text, end)
        if m:
            place, e2, p2 = _scan_name(text, m.end())
            if place and not p2 and _AFTER_PLACE.match(text, e2):
                return f"{name}, {place}", e2
    return name, end


# after "Woolsthorpe, Lincolnshire" the clause must end or carry on with
# another detail - otherwise the second name begins a new clause
_AFTER_PLACE = re.compile(
    r"\s*(?:$|[.;:,]|\s(?:and|in|on|at|to|from|where|which|who|when|"
    r"after|before|aged|until|as|near)\b)")


def _org_like(name: str, prep: str) -> bool:
    """Is this an employer? "Thomas Edison" is a person, not a place of
    work; "Bell Labs", "CERN", "the Swiss Patent Office" are."""
    words = [w for w in name.split() if w not in ("the", "The")]
    if any(w.strip(",") in _ORG_WORDS for w in words):
        return True
    if any(len(w) >= 2 and w.isalpha() and w.isupper() for w in words):
        return True
    return prep == "at" and len(words) == 1 and not _is_country(name)


def _names_list(text: str, pos: int) -> tuple[list[str], int]:
    """"Pierre Curie" / "Vincenzo Galilei and Giulia Ammannati"."""
    names = []
    for _ in range(3):
        name, end = _scan_obj(text, pos, allow_the=False)
        if not name or name.split()[0] in _NOT_SUBJECT:
            break
        names.append(name)
        pos = end
        m = _AND_NAME.match(text, pos)
        if not m:
            break
        pos = m.end()
    return names, pos


# ─── places ─────────────────────────────────────────────────────────

_REGION_OF = re.compile(
    r"\s+(?:region|province|state|county|department|district|prefecture|"
    r"canton|area|part)\s+of\s+")


def _strip_direction(t: str) -> str:
    w = t.split(" ", 1)
    if len(w) == 2 and w[0] in _DIRECTIONS:
        return w[1]
    return t


def _place_pp(text: str) -> list[tuple[str, str]]:
    """What follows "is a city in": "Tuscany, Italy" -> located_in
    Tuscany + country Italy; "the Veneto region of northern Italy" ->
    located_in Veneto + country Italy."""
    t = _strip_direction(text.lstrip())
    name, end, poss = _scan_name(t, 0, allow_the=True)
    if not name or poss:
        return []
    m = _REGION_OF.match(t, end)
    if not m and not _NAME_END.match(t, end):
        return []
    if m:
        t2 = _strip_direction(t[m.end():])
        name2, _ = _scan_obj(t2, 0)
        region = re.sub(r"^the\s+", "", name)
        if not name2:
            return [("located_in", region)]
        return _place_rels([region, name2])
    names = [name]
    while len(names) < 3:
        m = _COMMA_SP.match(t, end)
        if not m:
            break
        n2, e2, p2 = _scan_name(t, m.end(), allow_the=True)
        if not n2 or p2 or not _AFTER_PLACE.match(t, e2):
            break
        names.append(n2)
        end = e2
    return _place_rels(names)


def _place_rels(names: list[str]) -> list[tuple[str, str]]:
    rels = []
    for i, n in enumerate(names):
        if i == len(names) - 1 and _is_country(n):
            rels.append(("country", n))
        else:
            rels.append(("located_in", n))
    return rels


def _region_after(text: str, end: int) -> tuple[str | None, int]:
    """", Lincolnshire" after a birthplace - only when it is a bare name
    that ends the clause (", then part of ..." and ", Galileo has ..."
    are not regions)."""
    m = _COMMA_SP.match(text, end)
    if not m:
        return None, end
    name, e2, poss = _scan_name(text, m.end(), allow_the=True)
    if not name or poss or not _AFTER_PLACE.match(text, e2):
        return None, end
    return name, e2


# ─── born / died ────────────────────────────────────────────────────

_EV_DATE = re.compile(rf"\s*,?\s*(?:on\s+)?(?P<d>(?:(?:c\.|ca\.|circa)\s*)?"
                      rf"(?:{_DMY}|{_MDY}))(?:\s*(?:BC|BCE|AD|CE)\b)?")
_EV_YEAR = re.compile(rf"\s*,?\s*in\s+(?P<d>(?:{_MONTH}\s+)?{_YEAR})")
_EV_BARE_YEAR = re.compile(rf"\s+(?P<d>(?:c\.\s*)?{_YEAR})(?=\s*(?:[.;,]|$))")
_EV_PLACE = re.compile(r"\s*,?\s*(?:in|at)\s+")
_EV_SKIP = re.compile(r"\s+and\s+(?:raised|educated|grew\s+up)\b")
_EV_ASIDE = re.compile(r",\s+in\s+(?P<r>(?:the\s+)?[A-Z][^,.;]{0,60})"
                       r"(?:[,.;]|$)")


def _life_event(event: str, text: str
                ) -> list[tuple[str | None, str, str, tuple]]:
    """Facts from what follows "born" / "died": "in Pisa", "on 15
    February 1564", "in 1564", "in Woolsthorpe, Lincolnshire". Returns
    (subject or None for the clause subject, relation, value, quals)."""
    facts: list[tuple[str | None, str, str, tuple]] = []
    pos = 0
    got_place = got_date = False
    for step in range(4):
        rest = text[pos:]
        m = _EV_SKIP.match(rest)
        if m:
            pos += m.end()
            continue
        if not got_date:
            m = _EV_DATE.match(rest) or _EV_YEAR.match(rest) or (
                _EV_BARE_YEAR.match(rest) if step == 0 else None)
            if m:
                when = _parse_date(m.group("d"))
                if when:
                    facts += [(None, r, v, q)
                              for r, v, q in _date_facts(event, when)]
                    got_date = True
                    pos += m.end()
                    continue
        if not got_place:
            m = _EV_PLACE.match(rest)
            if m:
                name, end = _scan_obj(rest, m.end())
                if name:
                    facts.append((None, f"{event}_in", name, ()))
                    region, end = _region_after(rest, end)
                    if region:
                        rel = "country" if _is_country(region) \
                            else "located_in"
                        facts.append((_canon(name), rel, region, ()))
                    got_place = True
                    pos += end
                    continue
        elif facts:
            # "born in Ulm, in the Kingdom of Württemberg, on ..."
            m = _EV_ASIDE.match(rest)
            if m:
                region, _ = _scan_obj(m.group("r"), 0)
                place = next((v for s_, r, v, _ in facts
                              if s_ is None and r.endswith("_in")), None)
                if region and place and len(region) == len(
                        m.group("r").strip()):
                    rel = "country" if _is_country(region) else "located_in"
                    facts.append((_canon(place), rel, region, ()))
                    pos += m.end() - 1
                    continue
        break
    return facts


# ─── predicates ─────────────────────────────────────────────────────

# adverbs that may sit inside a verb phrase ("was later awarded", "is
# best known for") without changing what it states
_ADV_WORDS = frozenset("""
also later then first eventually subsequently briefly initially famously
primarily mainly mostly largely previously once originally formerly now
currently chiefly best well widely especially still again soon afterwards
thereafter both together already jointly most internationally formally
officially successfully
""".split())
_ADV = rf"(?:(?:{'|'.join(sorted(_ADV_WORDS))})\s+)*"
_LEAD_ADV_VP = re.compile(_ADV)
_VERBS = (r"was|is|were|are|became|becomes|has|had|have|studied|taught|"
          r"lectured|died|discovered|invented|developed|formulated|devised|"
          r"wrote|composed|painted|founded|co-founded|established|received|"
          r"won|earned|married|lived|moved|settled|worked|graduated|"
          r"attended|joined|published|orbits|lies|remains|remained|served|"
          r"left|returned|spent|made|designed|built|led|created|directed|"
          r"produced|read|co-discovered|co-invented|shared|resided")
_VERB_START = re.compile(rf"{_ADV}(?:{_VERBS})\b")
# "He was born in Ulm and died in Princeton" -> two verb phrases
_VP_SPLIT = re.compile(
    rf",?\s+(?:and|but)\s+(?:(?:later|also|then|subsequently|eventually|"
    rf"afterwards|finally)\s+)?(?=(?:{_VERBS})\b)")

_BORN_VP = re.compile(r"(?:was\s+|is\s+)?born\b")
_DIED_VP = re.compile(r"died\b")
_KNOWN_FOR = re.compile(
    rf"(?:is|was|are|were|became|becomes|remains|remained|has\s+been|"
    rf"had\s+been)\s+{_ADV}(?:known|famous|renowned|noted|celebrated|"
    rf"remembered|famed|recogni[sz]ed|notable)\s+for\s+")
_KNOWN_AS = re.compile(
    rf"(?:is|was|are|were)\s+{_ADV}(?:known|referred\s+to)\s+as\s+")
_CALLED = re.compile(
    rf"(?:is|was|are|were|has\s+been|have\s+been|had\s+been)\s+{_ADV}"
    rf"(?:often\s+|sometimes\s+|commonly\s+|popularly\s+|frequently\s+)?"
    rf"(?:called|nicknamed|dubbed)\s+")
_CAPITAL_OF = re.compile(
    rf"(?:is|was)\s+{_ADV}the\s+capital(?:\s+city)?(?:\s+and\s+(?:the\s+)?"
    rf"largest\s+city)?\s+of\s+(?=(?:the\s+)?[A-Z])")
_LOCATED = re.compile(
    rf"(?:is|was|lies|lay)\s+{_ADV}(?:located|situated)?\s*(?:in|on|within)"
    rf"\s+(?=(?:the\s+)?[A-Z])")
_LOCATED_STRICT = re.compile(
    rf"(?:(?:is|was)\s+{_ADV}(?:located|situated)|lies|lay)\s+"
    rf"(?:in|on|within)\s+(?=(?:the\s+)?[A-Z])")
_PART_OF = re.compile(
    rf"(?:is|was|forms|formed)\s+{_ADV}(?:an?\s+)?(?:integral\s+)?part\s+of"
    rf"\s+(?=(?:the\s+)?[A-Z])")
_HQ = re.compile(
    rf"(?:is|was|are|were)\s+{_ADV}(?:headquartered|based)\s+in\s+"
    rf"(?=(?:the\s+)?[A-Z])")
_MEMBER = re.compile(
    rf"(?:(?:is|was|were|are|became|becomes)\s+{_ADV}(?:an?\s+)"
    rf"(?:founding\s+|life\s+|honorary\s+|full\s+|foreign\s+|permanent\s+)?"
    rf"|(?:was|were)\s+{_ADV}elected\s+(?:as\s+)?(?:an?\s+)?"
    rf"(?:foreign\s+|honorary\s+)?)(?:member|Member|fellow|Fellow)\s+of\s+"
    rf"(?=(?:the\s+)?[A-Z])")
_JOINED = re.compile(r"joined\s+(?=(?:the\s+)?[A-Z])")
_PROFESSOR = re.compile(
    rf"(?:was|became|is|served\s+as)\s+{_ADV}(?:(?:appointed|named|made|"
    rf"elected)\s+(?:as\s+)?)?(?:an?\s+|the\s+)?(?:[A-Z]?[a-z]+\s+){{0,2}}?"
    rf"(?:professor|lecturer|Professor|Lecturer)(?:\s+of\s+[A-Z]?[a-z]+"
    rf"(?:\s+(?:and\s+)?[A-Z]?[a-z]+)?)?\s+at\s+(?=(?:the\s+)?[A-Z])")
_CHILD_OF = re.compile(
    rf"(?:was|is)\s+{_ADV}(?:the|a|an)\s+(?:(?:eldest|youngest|only|second|"
    rf"first|third|fourth|fifth|illegitimate|elder|younger|oldest)\s+)?"
    rf"(?:son|daughter|child)\s+of\s+(?=[A-Z])")
_FOUNDED_PASSIVE = re.compile(
    rf"(?:was|were)\s+{_ADV}(?:founded|established|chartered|incorporated|"
    rf"formed)\b")
_PUBLISHED = re.compile(
    rf"(?:was|were)\s+{_ADV}published\s+in\s+(?P<y>{_YEAR})")
_BY_AGENT = re.compile(
    rf"(?:was|were)\s+{_ADV}(?P<v>discovered|invented|developed|written|"
    rf"composed|painted)\s+(?:in\s+(?P<y>\d{{3,4}})\s+)?by\s+(?=[A-Z])")
_EDUCATED = re.compile(
    rf"(?:(?:was|were)\s+{_ADV})?(?:educated|trained|schooled)\s+at\s+"
    rf"(?=(?:the\s+)?[A-Z])")
_STUDIED = re.compile(
    r"(?:studied|read)\s+(?:(?!(?:at|in|under|with)\b)[a-z]+,?\s+){0,4}"
    r"(?:at|in)\s+(?=(?:the\s+)?[A-Z])")
_GRADUATED = re.compile(
    rf"graduated\s+{_ADV}(?:with\s+[^,]{{1,40}}?\s+)?from\s+"
    rf"(?=(?:the\s+)?[A-Z])")
_ATTENDED = re.compile(r"attended\s+(?=(?:the\s+)?[A-Z])")
_TAUGHT = re.compile(
    r"(?:taught|lectured)\s+(?:(?!(?:at|in)\b)[a-z]+,?\s+){0,4}(?:at|in)\s+"
    r"(?=(?:the\s+)?[A-Z])")
_MORE_ORGS = re.compile(
    r"\s*,?\s+(?:and|or)\s+(?:(?:later|then|subsequently|also|afterwards)"
    r"\s+)?(?:at|in)\s+(?=(?:the\s+)?[A-Z])")
_WORKED_AS = re.compile(rf"worked\s+{_ADV}as\s+(?:an?\s+)")
_WORKED_AT = re.compile(
    rf"worked\s+{_ADV}(?P<prep>at|for|in)\s+(?=(?:the\s+)?[A-Z])")
_MARRIED = re.compile(
    rf"(?:married|wed|(?:was|were)\s+{_ADV}married\s+to)\s+(?=[A-Z])")
_AWARD = re.compile(
    rf"(?:received|won|earned|shared|collected|(?:was|were)\s+{_ADV}"
    rf"(?:awarded|granted|given|presented\s+with))\s+")
_MADE = re.compile(
    r"(?P<v>(?:co-)?(?:discovered|invented|developed|formulated|devised))"
    r"\s+(?!(?:that|how|what|why|whether)\b)")
_MADE_REL = {"discovered": "discovered", "invented": "invented",
             "developed": "developed", "formulated": "developed",
             "devised": "developed"}
_WORKS = re.compile(
    r"(?:co-)?(?P<v>wrote|composed|painted|authored|penned)\s+"
    r"(?!(?:about|on|that|for|to|in|of|with|under|as|extensively|several|"
    r"many|numerous|over|more|some|a|an|his|her|their|music|poetry|books|"
    r"letters|articles|papers)\b)")
_WORKS_REL = {"wrote": "wrote", "authored": "wrote", "penned": "wrote",
              "composed": "composed", "painted": "painted"}
_AUTHOR_OF = re.compile(
    rf"(?:is|was)\s+{_ADV}the\s+(?P<v>author|composer|painter)\s+of\s+")
_FOUNDED = re.compile(
    r"(?:co-)?(?:founded|established|co-founded|set\s+up)\s+"
    r"(?=(?:the\s+)?[A-Z])")
_LIVED = re.compile(
    rf"(?:(?:lived|resided|settled)\s+{_ADV}(?:in|at)|(?:moved|emigrated|"
    rf"immigrated|relocated)\s+{_ADV}(?:back\s+)?to)\s+"
    rf"(?=(?:the\s+)?[A-Z])")
_APPROX = (r"(?:(?:about|approximately|around|roughly|nearly|almost|over|"
           r"more\s+than|less\s+than|fewer\s+than|some|an\s+estimated|"
           r"just\s+over|just\s+under|close\s+to|under)\s+)")
_NUMBER = (rf"(?P<num>{_APPROX}?\d[\d,]*(?:\.\d+)?"
           rf"(?:\s+(?:million|billion|thousand))?)")
_POPULATION = re.compile(
    rf"(?:has|had)\s+(?:a|an(?:\s+estimated)?)\s+(?:total\s+|estimated\s+|"
    rf"resident\s+)?population\s+of\s+{_NUMBER}(?:\s+(?:people|inhabitants|"
    rf"residents|persons))?(?:\s*,?\s*(?:as\s+of|in)\s+(?:[A-Za-z]+\s+)?"
    rf"(?P<y>\d{{4}}))?")
_POP_VALUE = re.compile(
    rf"{_NUMBER}(?:\s+(?:people|inhabitants|residents|persons))?"
    rf"(?:\s*,?\s*(?:as\s+of|in)\s+(?:[A-Za-z]+\s+)?(?P<y>\d{{4}}))?")
_AREA = re.compile(
    rf"(?:has\s+an?\s+(?:total\s+)?area\s+of|covers(?:\s+an\s+area\s+of)?)"
    rf"\s+(?P<v>{_APPROX}?\d[\d,]*(?:\.\d+)?\s+(?:square\s+(?:kilometres|"
    rf"kilometers|miles|metres|meters)|km2|km²|sq\s+mi|hectares|acres))")
_ORBITS = re.compile(r"orbits\s+(?=(?:the\s+)?[A-Z])")
_MOON_OF = re.compile(
    r"(?:is|was)\s+(?:an?\s+|the\s+(?:[a-z]+\s+){0,2})(?P<c>(?:natural\s+)?"
    r"(?:moon|satellite))\s+of\s+(?=(?:the\s+)?[A-Z])")
_SUPERLATIVE = re.compile(
    r"(?:is|was)\s+the\s+(?P<mods>(?:[a-z]+(?:-[a-z]+)?\s+){1,2})"
    r"(?P<cls>planet|dwarf\s+planet|moon|star|galaxy|city|town|village|"
    r"country|state|island|river|lake|mountain|ocean|sea|desert|port)\b")
_ASTRO = frozenset({"planet", "dwarf planet", "moon", "star", "galaxy"})
_COPULA = re.compile(
    rf"(?:is|was|are|were|became|becomes|remains|remained|has\s+been|"
    rf"had\s+been)\s+{_ADV}(?:an?)\s+")


def _split_vps(pred: str) -> list[str]:
    return [p for p in _VP_SPLIT.split(pred) if p.strip()]


def _read_predicate(c: _Clause, ctx: _Context, out: _Out,
                    when: dict) -> None:
    for vp in _split_vps(c.pred):
        _read_vp(c, vp, ctx, out, when)


def _read_vp(c: _Clause, vp: str, ctx: _Context, out: _Out,
             when: dict) -> None:
    vp = vp.strip()
    m = _LEAD_ADV_VP.match(vp)
    if m and m.end() and _VERB_START.match(vp, m.end()):
        vp = vp[m.end():]                # "later studied" -> "studied"
    for handler in _handlers_for(vp):
        if handler(c, vp, ctx, out, when):
            return


def _year_quals(own: str | None, when: dict) -> list[tuple[str, str]]:
    y = own or when.get("year")
    return [("year", y)] if y else []


def _h_born(c, vp, ctx, out, when) -> bool:
    for rx, event in ((_BORN_VP, "born"), (_DIED_VP, "died")):
        m = rx.match(vp)
        if not m:
            continue
        facts = _life_event(event, vp[m.end():])
        if not any(r.endswith(("_on", "_year")) for _, r, _, _ in facts) \
                and when.get("year"):
            facts.append((None, f"{event}_year", when["year"], ()))
        for subj, rel, val, q in facts:
            if subj is None:
                _emit(c, ctx, out, rel, val, q)
            elif not (c.kind or (ctx.kind if c.is_ctx else None)) in (
                    "place", "thing"):
                out.add(subj, rel, val, q)       # "Woolsthorpe -> Lincs."
        return True
    return False


def _h_known(c, vp, ctx, out, when) -> bool:
    m = _KNOWN_FOR.match(vp)
    if m:
        span = _cut(_cut(vp[m.end():], _HARD_END),
                    re.compile(r"\s+in\s+\d{3,4}\b"))
        if _thing_ok(span, loose=True):
            _emit(c, ctx, out, "known_for", span)
        return True
    m = _KNOWN_AS.match(vp) or _CALLED.match(vp)
    if m:
        span = _cut(vp[m.end():], _HARD_END)
        for alias in re.split(r"\s+or\s+", span):
            alias = alias.strip().strip("\"")
            if alias.startswith(("a ", "an ")) or not alias:
                continue
            if _title_ok(alias) or (alias.startswith("the ")
                                    and _thing_ok(alias[4:])):
                _emit(c, ctx, out, "alias", alias)
        return True
    return False


def _h_place_facts(c, vp, ctx, out, when) -> bool:
    kind = c.kind or (ctx.kind if c.is_ctx else None)
    m = _CAPITAL_OF.match(vp)
    if m:
        name, _ = _scan_obj(vp, m.end())
        if name and kind != "person":
            _emit(c, ctx, out, "capital_of", name)
            out.add(name, "capital", c.subject)
        return True
    m = (_LOCATED if c.is_ctx else _LOCATED_STRICT).match(vp)
    if m:
        if kind != "person":
            for rel, val in _place_pp(vp[m.end():]):
                _emit(c, ctx, out, rel, val)
        return True
    m = _PART_OF.match(vp)
    if m:
        name, _ = _scan_obj(vp, m.end())
        if name:
            _emit(c, ctx, out, "part_of", name)
        return True
    m = _HQ.match(vp)
    if m:
        name, _ = _scan_obj(vp, m.end())
        if name and kind != "person":
            _emit(c, ctx, out, "headquarters", name)
        return True
    m = _POPULATION.match(vp)
    if m:
        _emit(c, ctx, out, "population", m.group("num"),
              [("as_of", m.group("y"))] if m.group("y") else [])
        return True
    m = _AREA.match(vp)
    if m:
        _emit(c, ctx, out, "area", m.group("v"))
        return True
    m = _ORBITS.match(vp)
    if m:
        name, _ = _scan_obj(vp, m.end())
        if name:
            _emit(c, ctx, out, "orbits", name)
        return True
    m = _MOON_OF.match(vp)
    if m:
        name, _ = _scan_obj(vp, m.end())
        if name and kind != "person":
            _emit(c, ctx, out, "instance_of", m.group("c"))
            _emit(c, ctx, out, "orbits", name)
        return True
    m = _SUPERLATIVE.match(vp)
    if m:
        mods = m.group("mods").split()
        if kind != "person" and all(
                w in _ORDINALS or w.endswith("est") or w in ("most", "least",
                                                             "populous")
                or w.split("-")[-1].endswith("est") for w in mods):
            cls = m.group("cls")
            _emit(c, ctx, out, "instance_of", cls)
            _note_class(c, ctx, cls)
            rest = vp[m.end():]
            if re.match(r"\s+from\s+the\s+Sun\b", rest):
                _emit(c, ctx, out, "orbits", "the Sun")
            else:
                pm = re.match(r"\s+(?:in|of)\s+(?=(?:the\s+)?[A-Z])", rest)
                if pm and cls in _ASTRO:
                    name, _ = _scan_obj(rest, pm.end())
                    if name:
                        _emit(c, ctx, out, "part_of", name)
                elif pm:
                    for rel, val in _place_pp(rest[pm.end():]):
                        _emit(c, ctx, out, rel, val)
        return True
    return False


def _h_membership(c, vp, ctx, out, when) -> bool:
    m = _MEMBER.match(vp)
    if m:
        org, end = _scan_org(vp, m.end())
        if org:
            quals, _ = _period(vp[end:])
            _emit(c, ctx, out, "member_of", org, quals)
        return True
    m = _JOINED.match(vp)
    if m:
        org, _ = _scan_org(vp, m.end())
        if org and _org_like(org, "for"):
            _emit(c, ctx, out, "member_of", org)
        return True
    m = _PROFESSOR.match(vp)
    if m:
        org, end = _scan_org(vp, m.end())
        if org:
            quals, _ = _period(vp[end:])
            _emit(c, ctx, out, "taught_at", org, quals)
        return True
    return False


def _h_family(c, vp, ctx, out, when) -> bool:
    m = _CHILD_OF.match(vp)
    if m:
        names, _ = _names_list(vp, m.end())
        for n in names:
            _emit(c, ctx, out, "parent", n)
            out.add(n, "child", c.subject)
        return True
    m = _MARRIED.match(vp)
    if m:
        names, end = _names_list(vp, m.end())
        if len(names) == 1 and len(names[0].split()) <= 5:
            ym = re.match(r"\s+in\s+(\d{4})\b", vp[end:])
            quals = _year_quals(ym.group(1) if ym else None, when)
            before = len(out.facts)
            _emit(c, ctx, out, "spouse", names[0], quals)
            if len(out.facts) > before:      # marriage is symmetric
                out.add(names[0], "spouse", c.subject, quals)
        return True
    return False


def _h_passive(c, vp, ctx, out, when) -> bool:
    m = _BY_AGENT.match(vp)
    if m:
        names, _ = _names_list(vp, m.end())
        v = m.group("v")
        rel = {"written": "wrote", "composed": "composed",
               "painted": "painted"}.get(v, v)
        quals = _year_quals(m.group("y"), when)
        for n in names:
            out.add(n, rel, c.subject, quals)
            if v == "written":
                _emit(c, ctx, out, "author", n)
        return True
    m = _PUBLISHED.match(vp)
    if m:
        y = _parse_date(m.group("y"))
        if y:
            _emit(c, ctx, out, "published_year", y.year)
        return True
    m = _FOUNDED_PASSIVE.match(vp)
    if not m:
        return False
    if c.is_ctx and ctx.kind is None:
        ctx.kind = "thing"
    rest, got_year = vp[m.end():], False
    for _ in range(3):
        ym = re.match(rf"\s*,?\s*(?:in|on)\s+(?P<d>(?:{_DMY}|{_MDY}|"
                      rf"(?:{_MONTH}\s+)?{_YEAR}))", rest)
        if ym and not got_year:
            when_ = _parse_date(ym.group("d"))
            if when_:
                _emit(c, ctx, out, "founded_year", when_.year)
                got_year = True
            rest = rest[ym.end():]
            continue
        bm = re.match(r"\s*,?\s*by\s+(?=[A-Z])", rest)
        if bm:
            names, end = _names_list(rest, bm.end())
            for n in names:
                _emit(c, ctx, out, "founded_by", n)
                out.add(n, "founded", c.subject)
            rest = rest[end:]
            continue
        pm = re.match(r"\s*,?\s*in\s+(?=(?:the\s+)?[A-Z])", rest)
        if pm:
            _, end, _ = _scan_name(rest, pm.end(), allow_the=True)
            rest = rest[end:]
            continue
        break
    if not got_year and when.get("year"):
        _emit(c, ctx, out, "founded_year", when["year"])
    return True


def _h_education(c, vp, ctx, out, when) -> bool:
    for rx, rel in ((_EDUCATED, "educated_at"), (_STUDIED, "educated_at"),
                    (_GRADUATED, "educated_at"), (_ATTENDED, "educated_at"),
                    (_TAUGHT, "taught_at")):
        m = rx.match(vp)
        if not m:
            continue
        pos = m.end()
        for _ in range(3):
            org, end = _scan_org(vp, pos)
            if not org or (rx is _ATTENDED and not _org_like(org, "at")):
                break
            quals, used = _period(vp[end:])
            _emit(c, ctx, out, rel, org, quals)
            pos = end + used
            mm = _MORE_ORGS.match(vp, pos)
            if not mm:
                break
            pos = mm.end()
        return True
    return False


def _h_work(c, vp, ctx, out, when) -> bool:
    m = _WORKED_AS.match(vp)
    if m:
        items, nats, after = _parse_np(vp[m.end():])
        kind = c.kind or (ctx.kind if c.is_ctx else None)
        if kind not in ("place", "thing"):
            for it in items:
                _emit(c, ctx, out, "occupation", " ".join(it))
        am = re.match(r"\s*(?P<prep>at|for)\s+(?=(?:the\s+)?[A-Z])", after)
        if am:
            org, end = _scan_org(after, am.end())
            if org and _org_like(org, am.group("prep")):
                quals, _ = _period(after[end:])
                _emit(c, ctx, out, "worked_at", org, quals)
        return True
    m = _WORKED_AT.match(vp)
    if m:
        org, end = _scan_org(vp, m.end())
        if org and _org_like(org, m.group("prep")):
            quals, _ = _period(vp[end:])
            _emit(c, ctx, out, "worked_at", org, quals)
        return True
    m = _LIVED.match(vp)
    if m:
        name, end = _scan_obj(vp, m.end())
        if name:
            quals, _ = _period(vp[end:])
            quals = [("from", v) if k == "year" else (k, v)
                     for k, v in quals]
            _emit(c, ctx, out, "lived_in", name, quals)
        return True
    return False


def _h_award(c, vp, ctx, out, when) -> bool:
    m = _AWARD.match(vp)
    if not m:
        return False
    span = _cut(vp[m.end():], _HARD_END)
    items = _split_list(span)
    good = [_award_item(i) for i in items]
    if len(items) > 1 and not all(good):
        whole = _award_item(span)          # "Order of Arts and Letters"
        good = [whole] if whole else [g for g in good if g]
    for g in good:
        if g:
            value, year = g
            quals = _year_quals(year, when if len(good) == 1 else {})
            _emit(c, ctx, out, "award", value, quals)
    return True


def _award_item(item: str) -> tuple[str, str | None] | None:
    item = re.sub(r"\s+for\s+.*$", "", item.strip())
    value, year = _item_year(item)
    value = re.sub(r"^(?:a|an)\s+", "", value.strip())
    m = _AWARD_YEAR.match(value)             # "the 1921 Nobel Prize"
    if m:
        value, year = m.group(1) + m.group(3), year or m.group(2)
    return (value, year) if _award_ok(value) else None


_AWARD_YEAR = re.compile(r"^(the\s+)?(\d{4})\s+(.+)$")


def _h_made(c, vp, ctx, out, when) -> bool:
    m = _MADE.match(vp)
    if not m:
        return False
    rel = _MADE_REL[m.group("v").replace("co-", "")]
    span = _cut(vp[m.end():], _HARD_END)
    items = _split_list(span)
    for item in items:
        value, year = _item_year(item)
        value = _cut(value, _THING_END).strip()
        if _thing_ok(value):
            _emit(c, ctx, out, rel, value,
                  _year_quals(year, when if len(items) == 1 else {}))
    return True


def _h_works(c, vp, ctx, out, when) -> bool:
    m = _WORKS.match(vp) or _AUTHOR_OF.match(vp)
    if not m:
        return False
    v = m.group("v")
    rel = {"author": "wrote", "composer": "composed",
           "painter": "painted"}.get(v) or _WORKS_REL[v]
    span = _cut(vp[m.end():], _HARD_END)
    items = _split_list(span, works=True)
    for item in items:
        value, year = _item_year(item)
        value = _WORK_KIND.sub("", value.strip()).strip().strip("\"")
        if _title_ok(value):
            _emit(c, ctx, out, rel, value,
                  _year_quals(year, when if len(items) == 1 else {}))
    return True


def _h_founded(c, vp, ctx, out, when) -> bool:
    m = _FOUNDED.match(vp)
    if not m:
        return False
    org, end = _scan_org(vp, m.end())
    if org:
        ym = re.match(r"\s+in\s+(\d{3,4})\b(?![,\d])", vp[end:])
        year = ym.group(1) if ym else when.get("year")
        _emit(c, ctx, out, "founded", org, _year_quals(year, {}))
        out.add(org, "founded_by", c.subject)
        if year:
            out.add(org, "founded_year", year)
    return True


def _h_copula(c, vp, ctx, out, when) -> bool:
    m = _COPULA.match(vp)
    if not m:
        return False
    _copula_np(c, vp[m.end():], ctx, out, when)
    return True


_COPULAR = ("is", "was", "are", "were", "became", "becomes", "remains",
            "remained", "has", "had", "have")
# handlers in priority order, each with the first words it can match:
# reading a verb phrase tries only the handlers its first word allows
_VP_HANDLERS = (
    (_h_born, ("was", "is", "born", "died")),
    (_h_known, _COPULAR),
    (_h_place_facts, _COPULAR + ("lies", "lay", "forms", "formed", "covers",
                                 "orbits")),
    (_h_membership, _COPULAR + ("joined", "served")),
    (_h_family, ("was", "is", "were", "married", "wed")),
    (_h_passive, ("was", "were")),
    (_h_education, ("was", "were", "educated", "trained", "schooled",
                    "studied", "read", "graduated", "attended", "taught",
                    "lectured")),
    (_h_work, ("worked", "lived", "resided", "settled", "moved",
               "emigrated", "immigrated", "relocated")),
    (_h_award, ("received", "won", "earned", "shared", "collected", "was",
                "were")),
    (_h_made, ("discovered", "invented", "developed", "formulated",
               "devised", "co-discovered", "co-invented", "co-developed")),
    (_h_works, ("wrote", "composed", "painted", "authored", "penned",
                "co-wrote", "is", "was")),
    (_h_founded, ("founded", "established", "co-founded", "set")),
    (_h_copula, _COPULAR),
)
_HANDLERS_BY_WORD: dict[str, tuple] = {}
for _handler, _words in _VP_HANDLERS:
    for _w in _words:
        _HANDLERS_BY_WORD[_w] = _HANDLERS_BY_WORD.get(_w, ()) + (_handler,)

# Most encyclopedia sentences are copular ("was ...", "is ..."), and a
# copula alone would send them through ten handlers. The word after it
# (adverbs and "been" skipped) narrows that to the one or two that can
# match; an unlisted word falls back to every copular handler.
_AFTER_COPULA: dict[str, tuple] = {}
for _words, _handlers in (
        (("born", "died"), (_h_born,)),
        (("known", "famous", "renowned", "noted", "celebrated", "remembered",
          "famed", "recognized", "recognised", "notable", "referred",
          "called", "nicknamed", "dubbed"), (_h_known,)),
        (("the",), (_h_place_facts, _h_membership, _h_family, _h_works,
                    _h_copula)),
        (("a", "an"), (_h_place_facts, _h_membership, _h_family,
                       _h_copula)),
        (("located", "situated", "in", "on", "within", "part",
          "headquartered", "based"), (_h_place_facts,)),
        (("founded", "established", "chartered", "incorporated", "formed",
          "published", "discovered", "invented", "developed", "written",
          "composed", "painted"), (_h_passive,)),
        (("educated", "trained", "schooled"), (_h_education,)),
        (("awarded", "granted", "given", "presented"), (_h_award,)),
        (("married",), (_h_family,)),
        (("elected", "appointed", "named", "made"), (_h_membership,)),
        (("professor", "lecturer", "Professor", "Lecturer"),
         (_h_membership,))):
    for _w in _words:
        _AFTER_COPULA[_w] = _handlers


def _handlers_for(vp: str) -> tuple:
    words = vp.split(" ", 8)
    first = words[0]
    if first not in _COPULAR and first not in ("lies", "lay", "served"):
        return _HANDLERS_BY_WORD.get(first, ())
    for w in words[1:]:
        if w not in _ADV_WORDS and w != "been":
            return _AFTER_COPULA.get(w, _HANDLERS_BY_WORD.get(first, ()))
    return _HANDLERS_BY_WORD.get(first, ())


# ─── "X was an Italian astronomer, physicist and engineer" ─────────

_NP_TOKEN = re.compile(r"\S+")


@lru_cache(maxsize=4096)
def _parse_np(text: str) -> tuple[tuple[tuple[str, ...], ...],
                                  tuple[tuple[str, tuple], ...], str]:
    """Split a predicate noun phrase into nationalities and class items:
    "Polish and naturalised-French physicist and chemist in Paris" ->
    items [[physicist], [chemist]], nats [Polish, French(naturalised)],
    rest "in Paris". Items are 1-4 lower-case words; anything else ends
    the phrase."""
    toks = [(m.group(), m.start()) for m in _NP_TOKEN.finditer(text)]
    nats: list[tuple[str, tuple]] = []
    i, dropped = 0, 0
    while i < len(toks):
        w = toks[i][0]
        nat = _nationality(w)
        if nat and not w.endswith((".", ";", ":")):
            nats.append(nat)
            i += 1
            continue
        if w == "and" and nats and i + 1 < len(toks) \
                and _nationality(toks[i + 1][0]):
            i += 1
            continue
        # other capitalised modifiers ("German-born", "Renaissance",
        # "Nobel Prize-winning") describe, they are not the class
        if w[:1].isupper() and dropped < 3 and not w.endswith((",", ".",
                                                               ";", ":")):
            dropped += 1
            i += 1
            continue
        break
    items: list[list[str]] = []
    cur: list[str] = []
    end_char = toks[i][1] if i < len(toks) else len(text)
    while i < len(toks):
        w, start = toks[i]
        sep = w.endswith(",")
        stop_after = w.endswith((".", ";", ":", "!"))
        core = w.rstrip(",.;:!")
        if core == "and" and not sep and (cur or items):
            if cur:                          # "physicist and chemist"
                items.append(cur)
                cur = []
            i += 1                           # Oxford comma: ", and chemist"
            end_char = toks[i][1] if i < len(toks) else len(text)
            continue
        participle = core.endswith("ed") and core not in _ED_NOUNS and (
            not cur or (i + 1 < len(toks)
                        and toks[i + 1][0] in _PARTICIPLE_NEXT))
        if (not _CLASS_WORD.match(core) or core in _NP_STOP
                or (core.endswith("ly") and core not in _LY_NOUNS)
                or participle):
            end_char = start
            break
        cur.append(core)
        i += 1
        end_char = toks[i][1] if i < len(toks) else len(text)
        if len(cur) > 4:
            cur = []
            break
        if sep or stop_after:
            if not (sep and len(cur) == 1 and _adjective_like(cur[0])):
                items.append(cur)
            cur = []
            if stop_after:
                end_char = len(text)
                break
    if cur:
        items.append(cur)
    rest = text[end_char:] if end_char < len(text) else ""
    # "a student of Galileo", "a friend of": relational nouns mean
    # nothing without their "of" - drop them
    if items and rest.startswith("of ") and items[-1][-1] in _RELATIONAL:
        items.pop()
        rest = ""
    return tuple(tuple(it) for it in items), tuple(nats), rest


def _adjective_like(word: str) -> bool:
    """"high-level", "chemical", "influential" - modifiers, not kinds."""
    if word in _OCCUPATIONS or word in _PLACE_CLASSES or _is_occupation(
            (word,)):
        return False
    return "-" in word or word.endswith(("al", "ic", "ive", "ous", "ful",
                                         "less", "able", "ible"))


def _np_kind(items, hint: str | None) -> str:
    heads = [it[-1] for it in items]
    if hint == "person":
        return "person"
    if any(h in _PLACE_CLASSES or " ".join(it) in _PLACE_CLASSES
           for h, it in zip(heads, items)):
        return "place"
    if hint in ("place", "thing"):
        return hint
    if any(_is_occupation(it) for it in items):
        return "person"
    return "thing"


def _note_class(c: _Clause, ctx: _Context, cls: str) -> None:
    if c.is_ctx:
        ctx.classes.add(cls.split()[-1])


def _copula_np(c: _Clause, np_text: str, ctx: _Context, out: _Out,
               when: dict) -> None:
    items, nats, rest = _parse_np(np_text)
    hint = c.kind or (ctx.kind if c.is_ctx else None)
    kind = _np_kind(items, hint) if items else hint
    if not items:
        return
    if c.is_ctx and ctx.kind in (None, "thing") and kind in ("person",
                                                             "place"):
        ctx.kind = kind
    elif c.is_ctx and ctx.kind is None:
        ctx.kind = kind
    if kind == "person":
        for value, quals in nats:
            _emit(c, ctx, out, "nationality", value, quals)
        for it in items:
            _emit(c, ctx, out, "occupation", " ".join(it))
    else:
        for it in items:
            _emit(c, ctx, out, "instance_of", " ".join(it))
            _note_class(c, ctx, " ".join(it))
        if kind == "place":
            for value, _ in nats:
                country = _DEMONYMS.get(value)
                if country:
                    _emit(c, ctx, out, "country", country)
    rest = rest.lstrip()
    pm = re.match(r"(?:located\s+|situated\s+)?(?:in|on)\s+"
                  r"(?=(?:the\s+)?[A-Z]|(?:northern|southern|eastern|"
                  r"western|central)\s)", rest)
    if pm and kind != "person":
        for rel, val in _place_pp(rest[pm.end():]):
            _emit(c, ctx, out, rel, val)
    rm = re.match(r"(?:who|that|which)\s+", rest)
    if rm:
        _read_predicate(_Clause(c.subject, c.is_ctx, rest[rm.end():],
                                c.kind or (kind if not c.is_ctx else None)),
                        ctx, out, when)


# ─── sentence-level forms ───────────────────────────────────────────

_PRONOUN_START = re.compile(r"(?P<p>He|She|It|They|he|she|it|they)\s+"
                            r"(?=[a-z])")
_GENERIC_START = re.compile(r"The\s+(?P<n>[a-z][a-z-]+)\s+(?=[a-z])")
_LEAD = re.compile(r"(?P<name>[^()]{1,120}?)\s*\((?P<paren>[^()]*(?:\([^()]*"
                   r"\)[^()]*)*)\)(?P<rest>.*)$")
_KNOWN_AS_APPOS = re.compile(
    r",\s*(?:(?:commonly|also|better|more\s+commonly|popularly|usually|"
    r"professionally|simply|often|mononymously)\s+)?(?:known|referred\s+to|"
    r"called)\s+(?:as\s+|by\s+(?:the|his|her)\s+(?:pen\s+|stage\s+)?name\s+)?"
    r"(?P<alias>[^,]+),\s*")
_ANY_APPOS = re.compile(r",\s*[^,]{1,80},\s*(?=(?:was|is)\b)")
_PARENS = re.compile(r"\s*\([^()]*\)")
_LEAD_PP = re.compile(
    r"(?:In|On|By|During|Around|From|Since|After|Before|Between|Until|"
    r"Throughout|At|Following|Upon|Shortly\s+after|Soon\s+after|Later\s+in|"
    r"Early\s+in|Late\s+in)\s+(?P<pp>[^,]{1,60}),\s+(?=\S)")
_LEAD_ADV = re.compile(
    r"(?:Later|Also|Then|Eventually|Subsequently|Afterwards|Initially|"
    r"Originally|Meanwhile|However|Moreover|Furthermore|Additionally|"
    r"Notably|Today|Currently|Now|Ultimately|Finally|Thereafter|Together|"
    r"Previously|Earlier|Famously|Similarly|Likewise|Nevertheless|"
    r"Nonetheless|Indeed|Instead|Thus|Hence|Consequently|Accordingly)\b,?"
    r"\s+(?=\S)")
_PERIOD_PP = re.compile(r"(?P<a>\d{4})\s+and\s+(?P<b>\d{4})$|"
                        r"(?P<c>\d{4})\s+(?:to|until)\s+(?P<d>\d{4})$")
_KIN = {"father": "parent", "mother": "parent", "parents": "parent",
        "husband": "spouse", "wife": "spouse", "spouse": "spouse",
        "son": "child", "daughter": "child", "sons": "child",
        "daughters": "child", "children": "child"}
_KIN_INVERSE = {"parent": "child", "child": "parent", "spouse": "spouse"}
_OWNED = re.compile(
    r"(?:(?:late|elder|younger|eldest|youngest|first|second|only)\s+)?"
    r"(?P<noun>father|mother|parents|husband|wife|spouse|sons?|daughters?|"
    r"children|capital(?:\s+city)?|official\s+languages?|national\s+"
    r"languages?|currency|population|founders?)\b")


def _tidy(text: str) -> str:
    if text.isascii():
        return _WS.sub(" ", _CITE.sub("", text)).strip()
    t = (text.replace("’", "'").replace("‘", "'")
         .replace("“", '"').replace("”", '"')
         .replace("\xa0", " "))
    t = _CITE.sub("", t)
    return _WS.sub(" ", t).strip()


def _upper_first(s: str) -> str:
    return s[:1].upper() + s[1:]


def _strip_lead_phrase(s: str) -> tuple[str, dict]:
    """"In 1895, she married ..." -> ("She married ...", {"year":
    "1895"}). Leading adverbs and place phrases are dropped too."""
    when: dict = {}
    for _ in range(2):
        m = _LEAD_PP.match(s)
        if m:
            pp = m.group("pp").strip()
            pp = re.sub(r"^the\s+year\s+", "", pp)
            d = _parse_date(pp)
            if d and not d.circa:
                when = {"year": d.year}
            else:
                pm = _PERIOD_PP.match(pp)
                if pm:
                    when = {"from": pm.group("a") or pm.group("c"),
                            "to": pm.group("b") or pm.group("d")}
            s = _upper_first(s[m.end():])
            continue
        m = _LEAD_ADV.match(s)
        if m:
            s = _upper_first(s[m.end():])
            continue
        break
    return s, when


def _resolve(name: str, ctx: _Context) -> tuple[str, bool]:
    """Map a subject name to (canonical name, is it the article
    subject?). Short names ("Galileo", "Curie") count only for people."""
    k = norm_key(name)
    if ctx.subject:
        if k in ctx.names:
            return ctx.subject, True
        if " " not in k and k in ctx.short and ctx.kind in (None, "person"):
            return ctx.subject, True
    return _canon(name), False


def _pronoun_subject(p: str, s: str, ctx: _Context,
                     out: _Out) -> str | None:
    """Who He/She/It/They is, or None when it cannot be the subject."""
    if not ctx.subject:
        return None
    p = p.lower()
    if p in ("he", "she"):
        if ctx.kind not in (None, "person"):
            return None
        if ctx.pronoun and ctx.pronoun != p:
            return None                  # "She" in a "he" article
        ctx.kind = "person"
        ctx.pronoun = p
        # the pronoun fact waits until this sentence yields a fact about
        # the subject - proof that "he" really was the subject
        if not ctx.pronoun_said:
            ctx.pending = p
        return ctx.subject
    if p == "it":
        if ctx.kind == "person" or _EXPLETIVE_IT.match(_upper_first(s)):
            return None
        if ctx.kind is None:
            ctx.kind = "thing"
        return ctx.subject
    if p == "they" and ctx.plural and ctx.kind != "person":
        return ctx.subject
    return None


def _split_aside(rest: str) -> tuple[str | None, str | None]:
    """", who was born in Ulm, studied at ..." -> ("who was born in Ulm",
    "studied at ..."): the aside ends at the first comma followed by a
    verb."""
    for m in re.finditer(r",\s+", rest[1:]):
        pos = 1 + m.end()
        if _VERB_START.match(rest, pos):
            inner = rest[1:1 + m.start()].strip()
            if len(inner.split()) <= 15:
                return inner, rest[pos:]
            return None, None
    return None, None


def _read_aside(c: _Clause, inner: str, ctx: _Context, out: _Out,
                when: dict) -> None:
    if inner.startswith("who "):
        _read_predicate(_Clause(c.subject, c.is_ctx, inner[4:], c.kind),
                        ctx, out, when)
    elif inner.startswith(("born ", "died ")):
        _read_vp(c, inner, ctx, out, when)
    elif inner.startswith(("a ", "an ")):
        _copula_np(c, inner.split(" ", 1)[1], ctx, out, when)
    else:
        m = re.match(r"(?:also\s+|better\s+|commonly\s+)?known\s+as\s+(.+)$",
                     inner)
        if m and _title_ok(m.group(1)):
            _emit(c, ctx, out, "alias", m.group(1))


_LIST_REST = re.compile(r"\s+(?:and|or|with|&|nor)\b")
_PLURAL_VERB = re.compile(r"(?:were|are)\b")


def _subject_of(s: str, ctx: _Context, out: _Out,
                when: dict) -> _Clause | None:
    """Find who the sentence is about and where its predicate starts."""
    m = _PRONOUN_START.match(s)
    if m:
        subj = _pronoun_subject(m.group("p"), s, ctx, out)
        if not subj:
            return None
        return _Clause(subj, True, s[m.end():])
    m = _GENERIC_START.match(s)
    if m:
        # "The city is known for ..." in an article about a city
        if ctx.subject and m.group("n") in ctx.classes:
            return _Clause(ctx.subject, True, s[m.end():])
        return None
    first = s.split(" ", 1)[0].rstrip(",")
    if first in _NOT_SUBJECT:
        return None
    name, end, poss = _scan_name(s, 0, allow_the=True)
    if not name or poss or name in ("The", "the"):
        return None
    subj, is_ctx = _resolve(name, ctx)
    c = _Clause(subj, is_ctx, "")
    rest = s[end:]
    if rest.startswith(","):
        inner, pred = _split_aside(rest)
        if pred is None:
            return None
        _read_aside(c, inner, ctx, out, when)
        rest = " " + pred
    if not rest.startswith(" ") or not rest[1:2].islower() \
            or _LIST_REST.match(rest):
        return None                      # "Newton and Leibniz ...": a list
    c.pred = rest.strip()
    if c.is_ctx and _PLURAL_VERB.match(c.pred):
        ctx.plural = True                # "The Beatles were ..."
    return c


def _read_lead(s: str, ctx: _Context, out: _Out, when: dict) -> bool:
    """The first sentence of a biography: "Name (born – died), known as
    X, was ...". The dates are the subject's only when the subject is a
    person (a war has dates too)."""
    m = _LEAD.match(s)
    if not m:
        return False
    life = _life_dates(m.group("paren"))
    name = m.group("name").strip()
    if not life or not _looks_like_full_name(name):
        return False
    rest = _PARENS.sub("", m.group("rest"))
    aliases: list[str] = []
    am = _KNOWN_AS_APPOS.match(rest)
    if am:
        aliases = [a.strip() for a in re.split(r"\s+or\s+(?:\w+\s+)?(?:as\s+)?",
                                               am.group("alias"))
                   if _title_ok(a.strip())]
        rest = rest[am.end():]
    else:
        am = _ANY_APPOS.match(rest)
        if am:
            rest = rest[am.end():]
    pred = rest.strip()
    if not re.match(r"(?:was|is)\b", pred):
        return False
    # whose lead is it?
    if ctx.subject and _lead_is_subject(name, aliases, ctx):
        subj, is_ctx = ctx.subject, True
    elif ctx.subject:
        subj, is_ctx = _canon(aliases[0] if aliases else name), False
    else:
        ctx.adopt(_canon(aliases[0] if aliases else name))
        subj, is_ctx = ctx.subject, True
    person = (is_ctx and ctx.kind == "person") or _np_says_person(pred)
    if not person:
        return False
    c = _Clause(subj, is_ctx, pred, None if is_ctx else "person")
    if is_ctx:
        ctx.kind = "person"
        ctx.names |= {norm_key(_canon(name))} | {norm_key(a)
                                                 for a in aliases}
    for rel, val, q in life:
        _emit(c, ctx, out, rel, val, q)
    for alias in [_canon(name)] + aliases:
        if norm_key(alias) != norm_key(subj):
            _emit(c, ctx, out, "alias", alias)
    _read_predicate(c, ctx, out, when)
    return True


def _lead_is_subject(name: str, aliases: list[str], ctx: _Context) -> bool:
    if norm_key(_canon(name)) in ctx.names or any(
            norm_key(a) in ctx.names for a in aliases):
        return True
    base = _DISAMBIG.sub("", ctx.subject or "")
    title_toks = _name_tokens(base)
    name_toks = _name_tokens(name)
    if len(title_toks) == 1:
        return title_toks <= name_toks
    # "Galileo di Vincenzo Bonaiuti de' Galilei" shares two words with
    # "Galileo Galilei"; his father "Vincenzo Galilei" shares only one
    return len(title_toks & name_toks) >= 2


def _np_says_person(pred: str) -> bool:
    m = re.match(r"(?:was|is)\s+(?:an?\s+)(?P<np>.+)$", pred)
    if not m:
        return False
    items, _, _ = _parse_np(m.group("np"))
    return any(_is_occupation(it) for it in items) and not any(
        it[-1] in _PLACE_CLASSES for it in items)


def _read_born_participle(s: str, ctx: _Context, out: _Out,
                          when: dict) -> bool:
    """"Born in Pisa, then part of the Duchy of Florence, Galileo has been
    called ..." - the birth facts belong to the main clause's subject."""
    if not s.startswith("Born "):
        return False
    facts = _life_event("born", s[4:])
    if not facts:
        return False
    for m in re.finditer(r",\s+", s):
        clause = _subject_of(_upper_first(s[m.end():]), ctx, out, when)
        if clause is None:
            continue
        for subj, rel, val, q in facts:
            if subj is None:
                _emit(clause, ctx, out, rel, val, q)
            else:
                out.add(subj, rel, val, q)
        _read_predicate(clause, ctx, out, when)
        return True
    return True


_POSS_PRON = re.compile(r"(?P<p>His|Her|Its|Their)\s+")


def _read_possessive(s: str, ctx: _Context, out: _Out, when: dict) -> bool:
    """"His father, Vincenzo Galilei, was a lutenist" (a fact about the
    father, plus the family link), "Its capital and largest city is
    Reykjavík", "Iceland's capital is ..."."""
    m = _POSS_PRON.match(s)
    if m:
        p = m.group("p").lower()
        owner = _pronoun_subject({"his": "he", "her": "she", "its": "it",
                                  "their": "they"}[p], "It x", ctx, out)
        if not owner:
            return bool(_OWNED.match(s, m.end()))
        owner_c = _Clause(owner, True, "")
        pos = m.end()
    else:
        if "'s " not in s[:90]:
            return False
        name, end, poss = _scan_name(s, 0, allow_the=True)
        if not (name and poss) or name.split()[0] in _NOT_SUBJECT:
            return False
        subj, is_ctx = _resolve(name, ctx)
        owner_c = _Clause(subj, is_ctx, "")
        pos = end + 2                        # past "'s"
        while pos < len(s) and s[pos] == " ":
            pos += 1
    om = _OWNED.match(s, pos)
    if not om:
        return True                          # "His work ..." - not ours
    noun = om.group("noun")
    rest = s[om.end():]
    rel = _KIN.get(noun)
    if rel:
        _read_kin(owner_c, rel, rest, ctx, out, when)
        return True
    _read_owned(owner_c, noun, rest, ctx, out)
    return True


def _read_kin(owner: _Clause, rel: str, rest: str, ctx: _Context,
              out: _Out, when: dict) -> None:
    am = re.match(r",\s+(?=[A-Z])", rest)
    if am:                              # "His father, Vincenzo Galilei, ..."
        names, end = _names_list(rest, am.end())
        if len(names) != 1:
            return
        after = _PARENS.sub("", rest[end:])
        pm = re.match(r",\s+", after)
        _emit(owner, ctx, out, rel, names[0])
        out.add(names[0], _KIN_INVERSE[rel], owner.subject)
        if pm and _VERB_START.match(after, pm.end()):
            _read_predicate(_Clause(_canon(names[0]), False,
                                    after[pm.end():], "person"),
                            ctx, out, when)
        return
    vm = re.match(r"\s+(?:was|is|were|are)\s+(?=[A-Z])", rest)
    if vm:                              # "His father was Vincenzo Galilei."
        names, end = _names_list(rest, vm.end())
        if names and re.match(r"\s*(?:[.;,]|$)", rest[end:]):
            for n in names:
                _emit(owner, ctx, out, rel, n)
                out.add(n, _KIN_INVERSE[rel], owner.subject)


def _read_owned(owner: _Clause, noun: str, rest: str, ctx: _Context,
                out: _Out) -> None:
    """Values of "Its capital / official language / currency /
    population / founder is ..." for the owner."""
    m = re.match(r"(?:\s+and\s+(?:the\s+)?largest\s+city)?\s+(?:is|are|was|"
                 r"were)\s+", rest)
    if not m:
        return
    val = rest[m.end():]
    if noun.startswith("capital"):
        name, _ = _scan_obj(val, 0, allow_the=False)
        if name:
            _emit(owner, ctx, out, "capital", name)
            out.add(name, "capital_of", owner.subject)
    elif "language" in noun:
        names, _ = _names_list(val, 0)
        for n in names:
            _emit(owner, ctx, out, "official_language", n)
    elif noun == "currency":
        span = _cut(val, _HARD_END).strip()
        if 1 <= len(span.split()) <= 4 and _thing_ok(span):
            _emit(owner, ctx, out, "currency", span)
    elif noun == "population":
        pm = _POP_VALUE.match(val)
        if pm:
            _emit(owner, ctx, out, "population", pm.group("num"),
                  [("as_of", pm.group("y"))] if pm.group("y") else [])
    elif noun.startswith("founder"):
        names, _ = _names_list(val, 0)
        for n in names:
            _emit(owner, ctx, out, "founded_by", n)
            out.add(n, "founded", owner.subject)


_RELNOUN = re.compile(
    r"The\s+(?P<noun>capital(?:\s+city)?|official\s+languages?|national\s+"
    r"languages?|currency|population)(?:\s+and\s+largest\s+city)?"
    r"(?:\s+of\s+(?P<owner>[^,]{1,60}?))?(?=\s+(?:is|are|was|were)\s)")


def _read_relnoun(s: str, ctx: _Context, out: _Out) -> bool:
    """"The official language of Italy is Italian.", "The capital is
    Rome." (in the Italy article)."""
    m = _RELNOUN.match(s)
    if not m:
        return False
    owner_txt = m.group("owner")
    if owner_txt:
        name, end, poss = _scan_name(owner_txt, 0, allow_the=True)
        if not name or poss or end < len(owner_txt.rstrip()):
            return True
        subj, is_ctx = _resolve(name, ctx)
        owner = _Clause(subj, is_ctx, "")
    elif ctx.subject and ctx.kind != "person":
        owner = _Clause(ctx.subject, True, "")
    else:
        return True
    _read_owned(owner, m.group("noun"), s[m.end():], ctx, out)
    return True


# ─── reading texts ──────────────────────────────────────────────────

_ANCHOR = re.compile(r"(?P<a>[^:]{1,80}?):\s+(?P<rest>\S.*)$", re.S)
_ANCHORABLE = re.compile(r"(?:He|She|It|They|His|Her|Its|Their|Born)\b")
_SENT_END = re.compile(r"[.!?]\s+(?=[A-Z\"(])")


def _strip_anchor(text: str, ctx: _Context) -> str:
    """Drop the "Title: " that lattice/growth.py writes in front of a
    sentence that does not name its article ("Galileo Galilei: He taught
    at ..."). The anchor tells us the subject when nothing else does."""
    m = _ANCHOR.match(text)
    if not m:
        return text
    anchor, rest = m.group("a").strip(), m.group("rest")
    if ctx.subject:
        if _resolve(anchor, ctx)[1]:
            return rest
        return text
    if _looks_like_title(anchor) and _ANCHORABLE.match(rest):
        ctx.adopt(anchor)
        return rest
    return text


def _sentences(text: str) -> list[str]:
    if len(text) < 60:
        return [text]
    parts, start = [], 0
    for m in _SENT_END.finditer(text):
        word = text[start:m.start() + 1].rsplit(None, 1)[-1]
        core = word.rstrip(".").lstrip("(").lower()
        if word.endswith(".") and (core in _ABBREV or len(core) == 1):
            continue                         # "c. 1564", "J. R. R."
        parts.append(text[start:m.start() + 1])
        start = m.end()
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _read_sentence(s: str, ctx: _Context, out: _Out) -> None:
    ctx.pending = None
    if not s or len(s) > 500:
        return
    if _skip_sentence(s):
        return
    s, when = _strip_lead_phrase(s)
    if "(" in s and _read_lead(s, ctx, out, when):
        return
    s = _PARENS.sub("", s).strip()
    if not s:
        return
    if _read_born_participle(s, ctx, out, when):
        return
    if _read_possessive(s, ctx, out, when):
        return
    if _read_relnoun(s, ctx, out):
        return
    clause = _subject_of(s, ctx, out, when)
    if clause:
        _read_predicate(clause, ctx, out, when)


def _read_text(text: str, ctx: _Context, out: _Out) -> None:
    body = _strip_anchor(_tidy(text or ""), ctx)
    for sent in _sentences(body):
        _read_sentence(sent, ctx, out)


@lru_cache(maxsize=4096)
def _source_title(source: str | None) -> str | None:
    """"wikipedia:Galileo_Galilei" -> "Galileo Galilei"."""
    if not source:
        return None
    for prefix in ("wikipedia:", "wiki:"):
        if source.startswith(prefix):
            t = source[len(prefix):].split("#", 1)[0]
            t = t.replace("_", " ").strip()
            return t or None
    return None


# ─── public API ─────────────────────────────────────────────────────

def extract_facts(text: str, source: str = "", memory_id: int | None = None,
                  created_at: str | None = None,
                  subject: str | None = None) -> list[Fact]:
    """Facts stated by one stored text (usually one sentence).

    subject is the entity the text is about when a sentence does not
    name it (a pronoun, "Born in Pisa, ..."); it defaults to the title of
    a "wikipedia:<Title>" source. Every Fact carries text, source,
    memory_id and created_at."""
    ctx = _Context.about(subject or _source_title(source))
    out = _Out()
    out.prov = (text or "", source or "", memory_id, created_at)
    _read_text(text, ctx, out)
    return out.facts


def extract_from_rows(rows: Iterable[tuple]) -> list[Fact]:
    """Facts from memory rows (memory_id, text, source, created_at) in
    reading order. Rows of the same article share context, so "He died
    in London." three rows after the lead is about the article's subject
    - and a "She" in his article is about somebody else and is skipped.
    Rows without an article source are read one by one."""
    out = _Out()
    contexts: dict[str, _Context] = {}
    for row in rows:
        mid, text, source, created = (tuple(row) + (None,) * 4)[:4]
        title = _source_title(source)
        if title:
            ctx = contexts.get(norm_key(title))
            if ctx is None:
                ctx = contexts[norm_key(title)] = _Context.about(title)
        else:
            ctx = _Context()
        out.prov = (text or "", source or "", mid, created)
        _read_text(text, ctx, out)
    return out.facts
