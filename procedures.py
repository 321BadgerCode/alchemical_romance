import random

ACTIONS = [
	"Calcinate", "Distill", "Sublimate", "Coagulate", "Ferment",
	"Fix", "Project", "Dissolve", "Conjoin", "Separate", "Incinerate"
]

SUBJECTS = [
	"the Red King", "the White Queen", "the Mercury of the Sages", "the Dragon's Blood",
	"the Salt of Saturn", "the Stone of the Wise", "the Primordial Waters"
]

TOOLS = [
	"the Philosopher's Flask", "a crucible of obsidian", "the Hermetic retort",
	"the vessel of transformation", "a lens of quartz", "the athanor"
]

ADVERBS = ["gently", "vigorously", "repeatedly", "until blackness appears", "at dawn", "under moonlight"]

def generate_alchemical_step():
	action = random.choice(ACTIONS)
	subject = random.choice(SUBJECTS)
	tool = random.choice(TOOLS)
	adverb = random.choice(ADVERBS)
	return f"{action} {subject} in {tool} {adverb}."

def generate_procedure(n=5):
	return [generate_alchemical_step() for _ in range(n)]