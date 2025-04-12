import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os

GLYPHS = ["☉", "☽", "♂", "♀", "☿", "♃", "♄", "🜁", "🜂", "🜃", "🜄", "🜅", "🜍", "🜔", "🝍"]
FONT_PATH = "./Symbola.ttf"
IMAGE_SIZE = random.choice([64, 128])
LATENT_DIM = random.choice([8, 16, 32])
HIDDEN_DIM = random.choice([32, 64, 128])
EPOCHS = random.randint(10, 30)
UPDATE_RATE = EPOCHS // 10
LR = random.uniform(.01, 1)
OUTPUT_FILE = "./nn_fused_symbol.png"
NUM_INPUT_GLYPHS = random.randint(2, 5)
SEED = random.randint(0, 1000)
MODEL_FILE = "./autoencoder_model.npz"

def render_glyph(glyph, font):
	img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
	draw = ImageDraw.Draw(img)
	bbox = draw.textbbox((0, 0), glyph, font=font)
	w = bbox[2] - bbox[0]
	h = bbox[3] - bbox[1]
	pos = ((IMAGE_SIZE - w) // 2, (IMAGE_SIZE - h) // 2)
	draw.text(pos, glyph, font=font, fill=255)
	return np.array(img).astype(np.float32) / 255.0

class Autoencoder:
	def __init__(self, input_dim, latent_dim, hidden_dim):
		self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
		self.b1 = np.zeros((hidden_dim, 1))
		self.W2 = np.random.randn(latent_dim, hidden_dim) * 0.01
		self.b2 = np.zeros((latent_dim, 1))

		self.W3 = np.random.randn(hidden_dim, latent_dim) * 0.01
		self.b3 = np.zeros((hidden_dim, 1))
		self.W4 = np.random.randn(input_dim, hidden_dim) * 0.01
		self.b4 = np.zeros((input_dim, 1))

	def encode(self, x):
		z1 = np.tanh(self.W1 @ x + self.b1)
		z2 = np.tanh(self.W2 @ z1 + self.b2)
		return z2

	def decode(self, z, noise_factor=0.1):
		z_noisy = z + np.random.randn(*z.shape) * noise_factor

		z3 = np.tanh(self.W3 @ z_noisy + self.b3)
		out = sigmoid(self.W4 @ z3 + self.b4)
		return out

	def forward(self, x):
		z1 = np.tanh(self.W1 @ x + self.b1)
		z2 = np.tanh(self.W2 @ z1 + self.b2)
		z3 = np.tanh(self.W3 @ z2 + self.b3)
		out = sigmoid(self.W4 @ z3 + self.b4)
		return out, (z1, z2, z3)

	def backward(self, x, out, z1, z2, z3):
		m = x.shape[1]
		d_out = 2 * (out - x) * out * (1 - out)

		dW4 = d_out @ z3.T / m
		db4 = np.sum(d_out, axis=1, keepdims=True) / m
		dz3 = self.W4.T @ d_out * (1 - z3 ** 2)

		dW3 = dz3 @ z2.T / m
		db3 = np.sum(dz3, axis=1, keepdims=True) / m
		dz2 = self.W3.T @ dz3 * (1 - z2 ** 2)

		dW2 = dz2 @ z1.T / m
		db2 = np.sum(dz2, axis=1, keepdims=True) / m
		dz1 = self.W2.T @ dz2 * (1 - z1 ** 2)

		dW1 = dz1 @ x.T / m
		db1 = np.sum(dz1, axis=1, keepdims=True) / m

		return dW1, db1, dW2, db2, dW3, db3, dW4, db4

	def update(self, grads, lr):
		dW1, db1, dW2, db2, dW3, db3, dW4, db4 = grads
		self.W1 -= lr * dW1
		self.b1 -= lr * db1
		self.W2 -= lr * dW2
		self.b2 -= lr * db2
		self.W3 -= lr * dW3
		self.b3 -= lr * db3
		self.W4 -= lr * dW4
		self.b4 -= lr * db4

	def train(self, X, epochs, lr, update_rate=UPDATE_RATE):
		for epoch in range(epochs):
			out, (z1, z2, z3) = self.forward(X)
			loss = np.mean((X - out) ** 2)
			grads = self.backward(X, out, z1, z2, z3)
			self.update(grads, lr)

	def save(self, filename=MODEL_FILE):
		np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
				 W3=self.W3, b3=self.b3, W4=self.W4, b4=self.b4)

	def load(self, filename=MODEL_FILE):
		data = np.load(filename)
		self.W1 = data['W1']
		self.b1 = data['b1']
		self.W2 = data['W2']
		self.b2 = data['b2']
		self.W3 = data['W3']
		self.b3 = data['b3']
		self.W4 = data['W4']
		self.b4 = data['b4']

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def train_model():
	np.random.seed(SEED)
	if not os.path.exists(FONT_PATH):
		print("Missing Symbola.ttf. Please download and place in script folder.")
		return

	font = ImageFont.truetype(FONT_PATH, 48)
	selected_glyphs = random.sample(GLYPHS, NUM_INPUT_GLYPHS)

	images = [render_glyph(g, font).flatten().reshape(-1, 1) for g in selected_glyphs]
	X = np.hstack(images)

	ae = Autoencoder(input_dim=IMAGE_SIZE * IMAGE_SIZE, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM)
	ae.train(X, epochs=EPOCHS, lr=LR)
	ae.save(MODEL_FILE)

def image_to_ascii(image, width=80, ascii_chars="@%#*+=-:. "):
	img = image.convert("L")
	original_width, original_height = img.size
	aspect_ratio = original_height / original_width
	height = int(width * aspect_ratio)
	img = img.resize((width, height))
	img = img.convert('L')

	pixels = img.getdata()
	ascii_str = ""
	for pixel_value in pixels:
		ascii_str += ascii_chars[pixel_value // 32]

	ascii_str = '\n'.join([ascii_str[i:i + width] for i in range(0, len(ascii_str), width)])

	return ascii_str

def generate_symbol(output_file=OUTPUT_FILE):
	np.random.seed(SEED)
	if not os.path.exists(FONT_PATH):
		print("Missing Symbola.ttf. Please download and place in script folder.")
		return

	font = ImageFont.truetype(FONT_PATH, 48)
	selected_glyphs = random.sample(GLYPHS, NUM_INPUT_GLYPHS)

	images = [render_glyph(g, font).flatten().reshape(-1, 1) for g in selected_glyphs]
	X = np.hstack(images)

	ae = Autoencoder(input_dim=IMAGE_SIZE * IMAGE_SIZE, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM)
	ae.load(MODEL_FILE)

	latents = [ae.encode(x) for x in images]
	fused_latent = sum(latents) / len(latents)

	output = ae.decode(fused_latent)
	output = output.reshape((IMAGE_SIZE, IMAGE_SIZE))

	output = (output * 255).astype(np.uint8)
	img = Image.fromarray(output, mode='L')
	img = img.convert("RGBA")
	img = img.rotate(90, expand=True)
	img.save(output_file)

	print(f"Symbols: {', '.join(selected_glyphs)}")
	ascii_img = image_to_ascii(img, 20)
	print(ascii_img)