import matplotlib.pyplot as plt

from attention import attention_weights, tokens


def main() -> None:
	plt.imshow(attention_weights, aspect="auto")
	plt.xticks(range(len(tokens)), tokens)
	plt.yticks(range(len(tokens)), tokens)
	plt.colorbar()
	plt.title("Attention Heatmap")
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()