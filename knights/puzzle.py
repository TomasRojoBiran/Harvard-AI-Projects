from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    Not(And(AKnight, AKnave)),  # cannot be knight and knave at the same time
    Or(AKnight, AKnave),  # will be Knight or Knave
    Implication(AKnight, And(AKnight, AKnave)),
    Implication(AKnave, Not(And(AKnight, AKnave))),
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    Not(And(AKnave, BKnave)),
    And(AKnave, BKnight),
    Implication(AKnave, BKnight),
    Implication(BKnave, AKnight),
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    Not(And(AKnave, AKnight)),
    Or(AKnight, AKnave),
    Not(And(BKnave, BKnight)),
    Or(BKnight, BKnave),
    Implication(AKnave, And(AKnave, BKnight)),
    Implication(BKnave, And(AKnight, BKnight)),
    Biconditional(AKnave, BKnight),
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    Not(And(AKnave, AKnight)),
    Or(AKnight, AKnave),
    Not(And(BKnave, BKnight)),
    Or(BKnight, BKnave),
    Not(And(CKnight, CKnave)),
    Or(CKnave, CKnight),
    Implication(AKnave, CKnight),
    Implication(CKnight, AKnight),
    Implication(AKnight, BKnave),
    Biconditional(CKnight, AKnight),
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3),
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
