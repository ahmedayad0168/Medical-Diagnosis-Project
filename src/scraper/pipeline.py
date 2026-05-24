from src.scraper.mayoclinic_scraper import MayoClinicScraper
import argparse

def run_scraper(output="data/raw/diseases.json", use_mongo=False):
    scraper = MayoClinicScraper(delay=1, use_mongo=use_mongo)
    scraper.run(output_csv=output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/diseases.json")
    parser.add_argument("--mongo", action="store_true")
    args = parser.parse_args()
    run_scraper(args.output, args.mongo)