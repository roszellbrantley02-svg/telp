"""
A small encyclopedia for testing the fact layer: lead paragraphs written
in the style of Wikipedia introductions (real facts, our own wording),
each tagged with the source Telp would give it after `telp learn`.

CORPUS is a list of (source, sentence) rows in reading order - exactly
what lands in the memory. EXPECTED lists facts a good extractor must
find; FORBIDDEN lists tempting misreadings it must not produce.
"""

ARTICLES: dict[str, list[str]] = {
    "Galileo Galilei": [
        "Galileo di Vincenzo Bonaiuti de' Galilei (15 February 1564 – 8 "
        "January 1642), commonly referred to as Galileo Galilei, was an "
        "Italian astronomer, physicist and engineer.",
        "Born in Pisa, then part of the Duchy of Florence, Galileo has been "
        "called the father of modern science.",
        "He studied medicine at the University of Pisa but left without a "
        "degree.",
        "He taught mathematics at the University of Padua from 1592 to "
        "1610.",
        "Galileo discovered the four largest moons of Jupiter in 1610.",
        "His father, Vincenzo Galilei, was a lutenist and music theorist.",
        "He died in Arcetri, near Florence.",
    ],
    "Isaac Newton": [
        "Sir Isaac Newton (25 December 1642 – 20 March 1727) was an English "
        "mathematician, physicist and astronomer.",
        "He was born in Woolsthorpe, Lincolnshire.",
        "Newton studied at Trinity College, Cambridge.",
        "He taught at the University of Cambridge as Lucasian Professor of "
        "Mathematics.",
        "Newton is known for the laws of motion and universal gravitation.",
        "He died in London.",
    ],
    "Johannes Kepler": [
        "Johannes Kepler (27 December 1571 – 15 November 1630) was a German "
        "astronomer and mathematician.",
        "He was born in Weil der Stadt.",
        "Kepler studied at the University of Tübingen.",
        "He taught mathematics in Graz.",
        "Kepler is known for his laws of planetary motion.",
        "He died in Regensburg.",
    ],
    "Nicolaus Copernicus": [
        "Nicolaus Copernicus (19 February 1473 – 24 May 1543) was a Polish "
        "mathematician and astronomer.",
        "He was born in Toruń.",
        "Copernicus studied at the University of Kraków and later at the "
        "University of Padua.",
        "He is known for the heliocentric model of the Solar System.",
        "He died in Frombork.",
    ],
    "Marie Curie": [
        "Marie Salomea Skłodowska-Curie (7 November 1867 – 4 July 1934), "
        "known as Marie Curie, was a Polish and naturalised-French physicist "
        "and chemist.",
        "She was born in Warsaw.",
        "She studied at the University of Paris.",
        "Curie discovered polonium and radium.",
        "She received the Nobel Prize in Physics in 1903 and the Nobel Prize "
        "in Chemistry in 1911.",
        "She married Pierre Curie in 1895.",
        "She died in Passy, Haute-Savoie.",
    ],
    "Albert Einstein": [
        "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born "
        "theoretical physicist.",
        "He was born in Ulm.",
        "Einstein studied at ETH Zurich.",
        "He developed the theory of relativity.",
        "He received the Nobel Prize in Physics in 1921.",
        "He died in Princeton, New Jersey.",
    ],
    "William Shakespeare": [
        "William Shakespeare (c. 23 April 1564 – 23 April 1616) was an "
        "English playwright and poet.",
        "He was born in Stratford-upon-Avon.",
        "Shakespeare wrote Hamlet, Macbeth and Romeo and Juliet.",
        "He married Anne Hathaway in 1582.",
        "He died in Stratford-upon-Avon.",
    ],
    "Ada Lovelace": [
        "Augusta Ada King, Countess of Lovelace (10 December 1815 – 27 "
        "November 1852), known as Ada Lovelace, was an English mathematician "
        "and writer.",
        "She was born in London.",
        "She is known for her work on Charles Babbage's Analytical Engine.",
        "She died in London.",
    ],
    "Nikola Tesla": [
        "Nikola Tesla (10 July 1856 – 7 January 1943) was a Serbian-American "
        "inventor and electrical engineer.",
        "He was born in Smiljan.",
        "Tesla worked for Thomas Edison in New York.",
        "He invented the induction motor.",
        "He died in New York City.",
    ],
    "Pisa": [
        "Pisa is a city in Tuscany, Italy.",
        "It is known for the Leaning Tower of Pisa.",
    ],
    "Padua": [
        "Padua is a city in the Veneto region of northern Italy.",
        "The University of Padua was founded in 1222.",
    ],
    "Italy": [
        "Italy is a country in Southern Europe.",
        "Its capital is Rome.",
        "The official language of Italy is Italian.",
    ],
    "Iceland": [
        "Iceland is a Nordic island country in the North Atlantic Ocean.",
        "Its capital and largest city is Reykjavík.",
        "Iceland has a population of about 390,000.",
    ],
    "Poland": [
        "Poland is a country in Central Europe.",
        "Its capital is Warsaw.",
    ],
    "Germany": [
        "Germany is a country in Central Europe.",
        "Its capital is Berlin.",
    ],
    "Jupiter": [
        "Jupiter is the fifth planet from the Sun and the largest in the "
        "Solar System.",
        "It is a gas giant.",
    ],
}

CORPUS: list[tuple[str, str]] = [
    (f"wikipedia:{title}", s)
    for title, sents in ARTICLES.items() for s in sents
]

# facts the extractor must find: (subject, relation, value).
# Matching is by norm_key (case/accents/leading "the" ignored).
EXPECTED: list[tuple[str, str, str]] = [
    ("Galileo Galilei", "born_in", "Pisa"),
    ("Galileo Galilei", "born_on", "15 February 1564"),
    ("Galileo Galilei", "born_year", "1564"),
    ("Galileo Galilei", "died_on", "8 January 1642"),
    ("Galileo Galilei", "died_year", "1642"),
    ("Galileo Galilei", "died_in", "Arcetri"),
    ("Galileo Galilei", "nationality", "Italian"),
    ("Galileo Galilei", "occupation", "astronomer"),
    ("Galileo Galilei", "occupation", "physicist"),
    ("Galileo Galilei", "occupation", "engineer"),
    ("Galileo Galilei", "educated_at", "University of Pisa"),
    ("Galileo Galilei", "taught_at", "University of Padua"),
    ("Galileo Galilei", "discovered", "the four largest moons of Jupiter"),
    ("Isaac Newton", "born_in", "Woolsthorpe"),
    ("Isaac Newton", "born_year", "1642"),
    ("Isaac Newton", "nationality", "English"),
    ("Isaac Newton", "occupation", "mathematician"),
    ("Isaac Newton", "educated_at", "Trinity College, Cambridge"),
    ("Isaac Newton", "taught_at", "University of Cambridge"),
    ("Isaac Newton", "died_in", "London"),
    ("Johannes Kepler", "born_in", "Weil der Stadt"),
    ("Johannes Kepler", "nationality", "German"),
    ("Johannes Kepler", "educated_at", "University of Tübingen"),
    ("Johannes Kepler", "taught_at", "Graz"),
    ("Johannes Kepler", "died_in", "Regensburg"),
    ("Nicolaus Copernicus", "born_in", "Toruń"),
    ("Nicolaus Copernicus", "educated_at", "University of Kraków"),
    ("Nicolaus Copernicus", "educated_at", "University of Padua"),
    ("Nicolaus Copernicus", "nationality", "Polish"),
    ("Marie Curie", "born_in", "Warsaw"),
    ("Marie Curie", "educated_at", "University of Paris"),
    ("Marie Curie", "discovered", "polonium"),
    ("Marie Curie", "discovered", "radium"),
    ("Marie Curie", "award", "Nobel Prize in Physics"),
    ("Marie Curie", "award", "Nobel Prize in Chemistry"),
    ("Marie Curie", "spouse", "Pierre Curie"),
    ("Albert Einstein", "born_in", "Ulm"),
    ("Albert Einstein", "educated_at", "ETH Zurich"),
    ("Albert Einstein", "developed", "theory of relativity"),
    ("Albert Einstein", "award", "Nobel Prize in Physics"),
    ("William Shakespeare", "born_in", "Stratford-upon-Avon"),
    ("William Shakespeare", "wrote", "Hamlet"),
    ("William Shakespeare", "wrote", "Macbeth"),
    ("William Shakespeare", "wrote", "Romeo and Juliet"),
    ("William Shakespeare", "spouse", "Anne Hathaway"),
    ("William Shakespeare", "occupation", "playwright"),
    ("Ada Lovelace", "born_in", "London"),
    ("Ada Lovelace", "occupation", "mathematician"),
    ("Nikola Tesla", "born_in", "Smiljan"),
    ("Nikola Tesla", "invented", "induction motor"),
    ("Pisa", "instance_of", "city"),
    ("Pisa", "located_in", "Tuscany"),
    ("Pisa", "country", "Italy"),
    ("Italy", "capital", "Rome"),
    ("Italy", "official_language", "Italian"),
    ("Iceland", "capital", "Reykjavík"),
    ("Poland", "capital", "Warsaw"),
    ("Germany", "capital", "Berlin"),
    ("University of Padua", "founded_year", "1222"),
]

# misreadings that must NOT be produced
FORBIDDEN: list[tuple[str, str, str]] = [
    # the father's job is not Galileo's
    ("Galileo Galilei", "occupation", "lutenist"),
    ("Galileo Galilei", "occupation", "music theorist"),
    # "Born in Pisa, then part of the Duchy of Florence" - not born in the duchy
    ("Galileo Galilei", "born_in", "Duchy of Florence"),
    # a near-place in a death sentence is not the place of death
    ("Galileo Galilei", "died_in", "Florence"),
    # Tesla didn't work at Thomas Edison (a person); see worked_at handling
    ("Nikola Tesla", "worked_at", "Thomas Edison"),
]
