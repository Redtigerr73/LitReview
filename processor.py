import pandas as pd
import argparse

class Processor:
    def remove_duplicates(self, csv_path):
        df = pd.read_csv(csv_path)
        n_rows = df.shape[0]
        df = df.drop_duplicates(subset=['title', 'url'], keep=False)
        df.to_csv(csv_path, index=False)
        n_removed = n_rows - df.shape[0]
        print(f"Removed {n_removed} duplicate articles from {csv_path}")

    def run(self, csv_path):
        self.remove_duplicates(csv_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process articles from a CSV file')
    parser.add_argument('--csv', type=str, help='Path to the CSV file')
    args = parser.parse_args()

    processor = Processor()
    processor.run(args.csv)

