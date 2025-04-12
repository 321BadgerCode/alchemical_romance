from PIL import Image, ImageDraw, ImageFont
import numpy as np
import hashlib
import random
import os
from PIL import ImageOps, ImageEnhance, ImageChops

OUTPUT_IMAGE = "fused_symbol.png"
FONT_PATH = "Symbola.ttf"
GLYPHS = ["☉", "☽", "♂", "♀", "☿", "♃", "♄", "🜁", "🜂", "🜃", "🜄", "🜅", "🜍", "🜔", "🝍"]
FONT_SIZE = 128
IMG_SIZE = 256
NUM_GLYPHS_TO_COMBINE = 3

def render_glyph(glyph, font):
	img = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
	draw = ImageDraw.Draw(img)
	bbox = draw.textbbox((0, 0), glyph, font=font)
	w = bbox[2] - bbox[0]
	h = bbox[3] - bbox[1]
	position = ((IMG_SIZE - w) // 2, (IMG_SIZE - h) // 2)
	draw.text(position, glyph, font=font, fill=255)
	return img

def fuse_images(images):
	base = images[0].copy()

	for img in images[1:]:
		mode = random.choice(["blend", "add", "multiply", "screen"])
		if mode == "blend":
			alpha = random.uniform(0.3, 0.7)
			base = Image.blend(base, img, alpha)
		elif mode == "add":
			base = ImageChops.add(base, img)
		elif mode == "multiply":
			base = ImageChops.multiply(base, img)
		elif mode == "screen":
			base = ImageChops.screen(base, img)

	angle = random.randint(0, 360)
	base = base.rotate(angle, expand=True)
	return base

def stylize(img):
	img = ImageOps.autocontrast(img)
	img = ImageOps.invert(img)
	img = img.convert("RGBA")
	r, g, b, a = img.split()
	a = ImageEnhance.Brightness(a).enhance(1.3)
	return Image.merge("RGBA", (r, g, b, a))

def generate_symbol():
	if not os.path.exists(FONT_PATH):
		print(f"Missing font {FONT_PATH}. Please download Symbola.ttf and place it in this directory.")
		return

	font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
	glyphs = random.sample(GLYPHS, NUM_GLYPHS_TO_COMBINE)
	print("Selected glyphs:", glyphs)

	glyph_images = [render_glyph(g, font) for g in glyphs]
	fused = fuse_images(glyph_images)
	fused = ImageOps.autocontrast(fused)
	fused = ImageOps.invert(fused)
	fused = fused.convert("RGBA")

	# Optional stylization (add glow, tint, etc.)
	# fused = stylize(fused)

	fused.save(OUTPUT_IMAGE)
	print(f"Saved fused symbol as: {OUTPUT_IMAGE}")

def generate_diagram(filename):
	generate_symbol()
	print(f"Diagram saved as {filename}")