''' Starting point for ANLP 2018 assignment 2: CKY parsing'''
import re
import cfg_fix
from cfg_fix import parse_grammar, Tree
from cky import CKY


def tokenise(tokenstring):
    '''Split a string into a list of tokens

    We treat punctuation as
    separate tokens, and split contractions into their parts.

    So for example "I'm leaving." --> ["I","'m","leaving","."]

    :type tokenstring: str
    :param tokenstring: the string to be tokenised
    :rtype: list(str)
    :return: the tokens found in tokenstring'''
    return re.findall(
        # We use three sub-patterns:
        #   one for words and the first half of possessives
        #   one for the rest of possessives
        #   one for punctuation
        r"[-\w]+|'\w+|[^-\w\s]+",
        tokenstring,
        re.U  # Use unicode classes, otherwise we would split
        # "são jaques" into ["s", "ão","jaques"]
    )


grammar = parse_grammar("""
S -> NP VP
NP -> Det Nom | Nom | NP PP
Det -> NP "'s"
Nom -> N SRel | N
VP -> Vi | Vt NP | VP PP
PP -> Prep NP
SRel -> Relpro VP
Det -> 'a' | 'the'
N -> 'fish' | 'frogs' | 'soup' | 'children' | 'books'
Prep -> 'in' | 'for'
Vt -> 'saw' | 'ate' | 'read'
Vi -> 'fish' | 'swim'
Relpro -> 'that'
""")

# Use this grammar for the rest of the assignment
grammar2 = parse_grammar([
    "S -> Sdecl '.' | Simp '.' | Sq '?' ",
    "Sdecl -> NP VP",
    "Simp -> VP",
    "Sq -> Sqyn | Swhadv",
    "Sqyn -> Mod Sdecl | Aux Sdecl",
    "Swhadv -> WhAdv Sqyn",
    "Sc -> Subconj Sdecl",
    "NP -> PropN | Pro | NP0 ",  # NP that allows no modification
    "NP0 -> NP1 | NP0 PP",
    "NP1 -> Det N2sc | N2mp | Sc",
    "N2sc -> Adj N2sc | Nsc | N3 Nsc",
    "N2mp -> Adj N2mp | Nmp | N3 Nmp",
    "N3 -> N | N3 N",
    "N -> Nsc | Nmp",
    "VP -> VPi | VPt | VPdt | Mod VP | VP Adv | VP PP",
    "VPi -> Vi",  # intransitive
    "VPt -> Vt NP",  # transitive
    "VPdt -> VPo PP",  # ditransitive, obligatory NP (obj.) & PP complements
    "VPdt -> VPio NP",  # ditransitive, obligatory NP (iobj.) & NP (obj)
    "VPo -> Vdt NP",  # direct object of ditransitive
    "VPio -> Vdt NP",  # indirect obj. part of dative-shifted ditransitive
    "PP -> Prep NP",
    "Det -> 'a' | 'the'",
    "Nmp -> 'salad' | 'mushrooms'",  # mass or plural nouns
    "Nsc -> 'book' | 'fork' | 'flight' | 'salad' | 'drawing'",  # singular count nouns
    "Prep -> 'to' | 'with'",
    "Vi -> 'ate'",  # intransitive
    "Vt -> 'ate' | 'book' | 'Book' | 'gave' | 'told'",  # transitive
    "Vdt -> 'gave' | 'told' ",  # ditransitive
    "Subconj -> 'that'",  # subordinating conjunction
    "Mod -> 'Can' | 'will'",  # modal verbs
    "Aux -> 'did' ",  # auxiliary verbs
    "WhAdv -> 'Why'",
    "PropN -> 'John' | 'Mary' | 'NYC' | 'London'",
    "Adj -> 'nice' | 'drawing'",
    "Pro -> 'you' | 'he'",
    "Adv -> 'today'"
])

print(grammar)  # the simpler grammar
chart = CKY(grammar)
# this illustrates tracing of a very simple sentence; feel free to try others.
chart.recognise(tokenise("the frogs swim"), True)
chart.pprint()

# build a chart with the larger grammar
chart2 = CKY(grammar2)

# Note, please do _not_ use the Tree.draw() method uncommented
# _anywhere in this file_ (you are encouraged to use it in preparing
# your report).

# The sentences to examine.
#
for s in ["John gave a book to Mary.",
          "John gave Mary a book.",
          "John gave Mary a nice drawing book.",
          "John ate salad with mushrooms with a fork.",
          "Book a flight to NYC.",
          "Can you book a flight to London?",
          "Why did John book the flight?",
          "John told Mary that he will book a flight today."]:
    print(s, chart2.recognise(tokenise(s)))

chart2.pprint()

# Task 5
# for s in [...]:
#     print(s, chart2.parse(tokenise(s)))
#     print(chart2.firstTree().pprint())



