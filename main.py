import random
from core import *
from latin import *
from procedures import *
from symbol_drawer import *

def capitalize(word):
	return word[0].upper() + word[1:]

def generate_alchemical_symbol():
	metal = random.choice(BASE_METALS)
	name = generate_pseudo_latin()
	procedure = generate_procedure()

	print(f"Name: {capitalize(name)}")
	print(f"Associated Metal: {metal.name} ({metal.planet}, {metal.glyph})")
	print("\nProcedure:")
	for step in procedure:
		print(f" - {step}")

	print()
	generate_symbol(f"{name}_symbol.png")

if __name__ == "__main__":
	train_model()
	generate_alchemical_symbol()