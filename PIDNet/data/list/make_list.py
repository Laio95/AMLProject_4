import argparse

def generate_file_list(output_path: str, count: int):
    with open(output_path, "w") as f:
        for i in range(count):
            line = f"Train/Rural/images_png/{i}.png\tTrain/Rural/masks_png/{i}.png\n"
            f.write(line)
    print(f"File generated at: {output_path} with {count} lines.")

def main():
    parser = argparse.ArgumentParser(description="Generate a list of image and mask paths.")
    parser.add_argument("output_path", type=str, help="Path to the output file")
    parser.add_argument("count", type=int, help="Number of lines to generate")

    args = parser.parse_args()
    generate_file_list(args.output_path, args.count)

if __name__ == "__main__":
    main()
