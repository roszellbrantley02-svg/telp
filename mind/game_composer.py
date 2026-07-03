"""autopilot/game_composer.py — generator for small terminal games.

Games are a different archetype from data apps:
  * data apps  = store + retrieve + display records  (CRUD)
  * games      = game loop + state + input + win/loss conditions

This module handles the games archetype.  It recognizes:
  - generic "build me a game" / "build me a gaming app"
  - specific games by name (hangman, tic-tac-toe, etc.)

Each game template is a complete, runnable terminal game.  The demo
block at the bottom of each file runs a non-interactive simulation
(scripted inputs) so the demo output is visible to the user even
though the real game uses input() interactively.

Currently supported:
  * guess_the_number   — pick 1-100, get hot/cold hints
  * rock_paper_scissors — vs random CPU, score-tracking
  * hangman            — random word, letter guesses
  * tic_tac_toe        — vs random CPU on 3x3 board
  * dice_roller        — roll NdM, useful for RPGs
  * coin_flip          — heads/tails toss, score-tracking
"""
from __future__ import annotations

import re
from typing import Optional


# ─── Game library ────────────────────────────────────────────────


# Each game: {name → {code, label, intents}}

_GUESS_NUMBER_CODE = '''"""A terminal "guess the number" game.

Pick a number 1-100; the computer tells you higher/lower each guess.

Run:  python guess.py
"""
import random
import sys

def guess_the_number(target=None, max_attempts=10, _inputs=None):
    """Run one round.  If `_inputs` is given, use it as a scripted
    input sequence instead of input() — used by the demo block."""
    if target is None:
        target = random.randint(1, 100)
    print(f"I'm thinking of a number between 1 and 100.")
    print(f"You have {max_attempts} attempts.")
    input_idx = 0
    for attempt in range(1, max_attempts + 1):
        if _inputs is not None:
            if input_idx >= len(_inputs):
                print("(out of scripted inputs)")
                return False
            raw = _inputs[input_idx]
            input_idx += 1
            print(f"Attempt {attempt}: {raw}")
        else:
            raw = input(f"Attempt {attempt}: ")
        try:
            guess = int(raw)
        except ValueError:
            print("  please enter a number")
            continue
        if guess == target:
            print(f"  yes! got it in {attempt}")
            return True
        elif guess < target:
            print("  higher")
        else:
            print("  lower")
    print(f"  out of attempts.  the number was {target}")
    return False

if __name__ == "__main__":
    try: guess_the_number()
    except (EOFError, KeyboardInterrupt): pass

# === Live demo (scripted inputs, target=42) ===
print("--- demo run: scripted guesses, target=42 ---")
guess_the_number(target=42, _inputs=["50", "25", "37", "43", "40", "42"])
'''

_RPS_CODE = '''"""Rock-paper-scissors vs random CPU.

Run:  python rps.py
"""
import random

CHOICES = ("rock", "paper", "scissors")
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}

def play_round(player, cpu=None):
    cpu = cpu or random.choice(CHOICES)
    print(f"  you: {player}    cpu: {cpu}")
    if player == cpu:
        print("  draw")
        return "draw"
    if BEATS[player] == cpu:
        print("  you win")
        return "win"
    print("  cpu wins")
    return "loss"

def main(n_rounds=5, _scripted=None):
    score = {"win": 0, "loss": 0, "draw": 0}
    for i in range(n_rounds):
        if _scripted is not None and i < len(_scripted):
            player, cpu = _scripted[i]
            print(f"round {i+1}:")
        else:
            print(f"round {i+1}: choose rock/paper/scissors")
            player = input("> ").strip().lower()
            cpu = None
        if player not in CHOICES:
            print("  invalid; skipping")
            continue
        result = play_round(player, cpu=cpu)
        score[result] += 1
    print(f"final: wins={score['win']}  losses={score['loss']}  "
          f"draws={score['draw']}")
    return score

if __name__ == "__main__":
    try: main()
    except (EOFError, KeyboardInterrupt): pass

# === Live demo (scripted plays) ===
print("--- demo round of rps ---")
main(n_rounds=4, _scripted=[
    ("rock",     "scissors"),
    ("paper",    "rock"),
    ("scissors", "rock"),
    ("rock",     "rock"),
])
'''

_HANGMAN_CODE = '''"""Hangman — guess letters in a hidden word.

Run:  python hangman.py
"""
import random

WORDS = [
    "python", "trader", "lattice", "memory", "vector",
    "telp", "hyperdimensional", "neural", "symbolic", "encoder",
]

def play(word=None, max_misses=6, _scripted_guesses=None):
    word = (word or random.choice(WORDS)).lower()
    guessed = set()
    misses = 0
    idx = 0
    while misses < max_misses:
        shown = " ".join(c if c in guessed else "_" for c in word)
        print(f"  {shown}     misses={misses}/{max_misses}")
        if all(c in guessed for c in word):
            print(f"  you got it: {word}")
            return True
        if _scripted_guesses is not None:
            if idx >= len(_scripted_guesses):
                print("  (out of scripted guesses)")
                return False
            g = _scripted_guesses[idx].lower()
            idx += 1
            print(f"  guess: {g}")
        else:
            g = input("  guess a letter: ").strip().lower()
        if not g or len(g) != 1 or not g.isalpha():
            print("  one letter please")
            continue
        if g in guessed:
            print(f"  already guessed {g!r}")
            continue
        guessed.add(g)
        if g in word:
            print(f"  {g!r} is in the word")
        else:
            misses += 1
            print(f"  no {g!r}")
    print(f"  out of misses. word was {word!r}")
    return False

if __name__ == "__main__":
    try: play()
    except (EOFError, KeyboardInterrupt): pass

# === Live demo (scripted guesses, word="python") ===
print("--- demo: word='python' ---")
play(word="python",
     _scripted_guesses=["e", "a", "p", "y", "t", "h", "o", "n"])
'''

_TIC_TAC_TOE_CODE = '''"""Tic-Tac-Toe vs random CPU on a 3x3 board.

Run:  python ttt.py
"""
import random

EMPTY = "."

def _print(board):
    for r in range(3):
        print("  " + " ".join(board[r*3 + c] for c in range(3)))
    print()

def _winner(board):
    lines = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6),
    ]
    for a,b,c in lines:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    if EMPTY not in board:
        return "draw"
    return None

def _cpu_move(board):
    empties = [i for i, v in enumerate(board) if v == EMPTY]
    return random.choice(empties)

def play(_scripted_moves=None):
    board = [EMPTY] * 9
    turn = "X"   # human is X
    move_idx = 0
    while True:
        _print(board)
        w = _winner(board)
        if w is not None:
            print(f"  result: {w}")
            return w
        if turn == "X":
            if _scripted_moves is not None and move_idx < len(_scripted_moves):
                mv = _scripted_moves[move_idx]
                move_idx += 1
                print(f"  you (X) play {mv}")
            else:
                raw = input("  your move (0-8): ").strip()
                try:
                    mv = int(raw)
                except ValueError:
                    continue
            if not (0 <= mv < 9) or board[mv] != EMPTY:
                print("  illegal")
                continue
            board[mv] = "X"
        else:
            mv = _cpu_move(board)
            print(f"  cpu (O) plays {mv}")
            board[mv] = "O"
        turn = "O" if turn == "X" else "X"

if __name__ == "__main__":
    try: play()
    except (EOFError, KeyboardInterrupt): pass

# === Live demo (scripted X moves; CPU random) ===
print("--- demo run ---")
random.seed(0)
play(_scripted_moves=[4, 0, 8, 2, 6])
'''

_DICE_CODE = '''"""Dice roller — useful for RPGs.  Roll NdM (e.g., 3d6).

Run:  python dice.py 3d6
"""
import random
import re
import sys

def roll(spec):
    m = re.fullmatch(r"(\\d+)d(\\d+)", spec)
    if not m:
        raise ValueError(f"bad spec: {spec!r}, want like '3d6'")
    n = int(m.group(1))
    sides = int(m.group(2))
    if n < 1 or sides < 2:
        raise ValueError("n>=1 and sides>=2")
    results = [random.randint(1, sides) for _ in range(n)]
    return {"spec": spec, "rolls": results, "total": sum(results)}

if __name__ == "__main__":
    spec = sys.argv[1] if len(sys.argv) > 1 else "1d20"
    r = roll(spec)
    print(f"{r['spec']}: {r['rolls']} = {r['total']}")

# === Live demo ===
random.seed(42)
print("--- demo rolls ---")
for s in ("1d20", "3d6", "4d6", "2d10", "1d100"):
    r = roll(s)
    print(f"  {r['spec']:>6}: {r['rolls']!r:<25}  total={r['total']}")
'''

_COIN_FLIP_CODE = '''"""Coin-flip game.  Call it: heads or tails.

Run:  python coin.py
"""
import random

def flip():
    return random.choice(("heads", "tails"))

def play(call, _result=None):
    result = _result or flip()
    print(f"  you called {call}    coin: {result}")
    if call == result:
        print("  win"); return True
    print("  loss"); return False

def session(n=5, _scripted=None):
    wins = 0
    for i in range(n):
        if _scripted is not None and i < len(_scripted):
            call, result = _scripted[i]
        else:
            call = input(f"round {i+1}: heads or tails? ").strip().lower()
            result = flip()
        if call not in ("heads", "tails"):
            print("  invalid"); continue
        if play(call, _result=result):
            wins += 1
    print(f"session: {wins}/{n} wins")

if __name__ == "__main__":
    try: session()
    except (EOFError, KeyboardInterrupt): pass

# === Live demo ===
random.seed(7)
print("--- demo flips ---")
session(n=5, _scripted=[
    ("heads", "heads"),
    ("tails", "heads"),
    ("heads", "heads"),
    ("tails", "tails"),
    ("heads", "tails"),
])
'''


GAMES = {
    "guess_the_number": {
        "code":   _GUESS_NUMBER_CODE,
        "label":  "guess-the-number — pick 1-100, hot/cold hints",
        "intents": [
            r"\bguess(?:ing)?\s+(?:the\s+)?number\b",
            r"\bnumber\s+guess",
            r"\bhigher\s+lower",
        ],
    },
    "rock_paper_scissors": {
        "code":   _RPS_CODE,
        "label":  "rock-paper-scissors vs CPU",
        "intents": [
            r"\brock\s*[- ]?\s*paper\s*[- ]?\s*scissors",
            r"\brps\b",
        ],
    },
    "hangman": {
        "code":   _HANGMAN_CODE,
        "label":  "hangman — guess letters in a hidden word",
        "intents": [
            r"\bhangman\b",
            r"\bword\s+guess",
        ],
    },
    "tic_tac_toe": {
        "code":   _TIC_TAC_TOE_CODE,
        "label":  "tic-tac-toe on a 3x3 grid vs CPU",
        "intents": [
            r"\btic[- ]?tac[- ]?toe",
            r"\bttt\b",
            r"\bnoughts?\s+(?:and|&)\s+crosses",
            r"\bx\s+(?:and|&)\s+o\s+game",
        ],
    },
    "dice_roller": {
        "code":   _DICE_CODE,
        "label":  "dice roller (NdM notation, useful for RPGs)",
        "intents": [
            r"\bdice\s+(?:roll|roller|game)",
            r"\broll\s+(?:some\s+)?dice",
            r"\bd&d\s+dice",
        ],
    },
    "coin_flip": {
        "code":   _COIN_FLIP_CODE,
        "label":  "coin-flip game — heads or tails",
        "intents": [
            r"\bcoin\s+(?:flip|toss|game)",
            r"\bflip\s+a\s+coin",
            r"\bheads\s+or\s+tails",
        ],
    },
}


# Default game when user says "a game" without specifying
_DEFAULT_GAME = "guess_the_number"


# ─── Intent detection ────────────────────────────────────────────


_GENERIC_GAME_RX = re.compile(
    r"\b(?:build|make|create|write)\s+(?:me\s+)?(?:a\s+|an\s+|some\s+)?"
    r"(\w+(?:[- ]\w+){0,2})?\s*(?:game|gaming(?:\s+app)?|terminal\s+game)\b",
    re.IGNORECASE,
)


def detect_game_intent(msg: str) -> Optional[str]:
    """Return the game key (e.g., 'hangman') or None.

    Priority:
      1. Specific game name appears in the message → that game.
      2. Generic "a game" / "gaming app" → default game.
      3. Otherwise None.
    """
    if not msg:
        return None
    # First: check each game's intent patterns (specific names win).
    for key, spec in GAMES.items():
        for pat in spec["intents"]:
            if re.search(pat, msg, re.IGNORECASE):
                return key

    # Second: generic "a game" request → default.
    if _GENERIC_GAME_RX.search(msg):
        return _DEFAULT_GAME

    return None


# ─── Public entry ────────────────────────────────────────────────


def try_compose_game(msg: str, run: bool = True) -> Optional[dict]:
    """Detect game intent + generate a game.

    Returns a result dict matching try_write_code's contract, or None.
    """
    key = detect_game_intent(msg)
    if key is None:
        return None
    spec = GAMES[key]
    code = spec["code"]
    result = {
        "code":       code,
        "label":      spec["label"],
        "template":   f"game:{key}",
        "ran":        False,
        "output":     "",
        "customized": False,
        "game":       key,
    }
    if run:
        from mind.code_writer import _safe_run
        ok, out = _safe_run(code, timeout=10.0)
        result["ran"]    = ok
        result["output"] = out
    return result


# ─── Self-test ────────────────────────────────────────────────────


def _self_test():
    cases = [
        "build me a game",
        "build me a gaming app",
        "make me a hangman game",
        "build a tic tac toe game",
        "build a rock paper scissors game",
        "let's play rps",
        "build me a dice roller",
        "make a coin flip game",
        "build me a guess the number game",
        "build me a noughts and crosses game",
        "build a note app",   # not a game
        "hey",                 # not a game
    ]
    for msg in cases:
        key = detect_game_intent(msg)
        print(f"  {key!r:<22}  ← {msg!r}")


if __name__ == "__main__":
    _self_test()
