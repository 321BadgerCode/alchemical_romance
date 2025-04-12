from collections import namedtuple

BaseMetal = namedtuple("BaseMetal", ["name", "planet", "glyph", "color", "properties"])

BASE_METALS = [
	BaseMetal("Lead", "Saturn", "♄", "gray", ["heavy", "mutable", "corrupt"]),
	BaseMetal("Tin", "Jupiter", "♃", "silver", ["resonant", "amplifying"]),
	BaseMetal("Iron", "Mars", "♂", "red", ["violent", "piercing"]),
	BaseMetal("Gold", "Sun", "☉", "gold", ["perfect", "radiant"]),
	BaseMetal("Copper", "Venus", "♀", "green", ["attractive", "harmonious"]),
	BaseMetal("Mercury", "Mercury", "☿", "silver", ["fluid", "volatile"]),
	BaseMetal("Silver", "Moon", "☽", "white", ["pure", "reflective"]),
]