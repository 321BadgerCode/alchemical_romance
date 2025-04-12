import random

LATIN_ROOTS = [
	"ign", "sulph", "aqua", "terra", "aer", "spirit", "corp", "vis", "aeth", "lumin"
]

LATIN_SUFFIXES = [
	"ium", "atus", "ensis", "or", "us", "is", "um", "ae", "i", "os"
]

PREFIXES = ["trans", "sub", "contra", "meta", "in", "ex", "per", "ultra"]

def generate_pseudo_latin():
	root = random.choice(LATIN_ROOTS)
	suffix = random.choice(LATIN_SUFFIXES)
	if random.random() < 0.5:
		root = random.choice(PREFIXES) + root
	return root + suffix