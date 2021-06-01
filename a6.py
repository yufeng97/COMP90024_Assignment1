# -*- coding: utf-8 -*-
import argparse
import json
import datetime
import re
import mmap
from mpi4py import MPI


TWITTER_FILE = None
GRID_FILE = None
SCORE_FILE = None


class Utils:
    coordinates_pattern = re.compile(r'"coordinates":\[(.*?)]')
    text_pattern = re.compile(r'"text":"(.*?)","loc')
    punctuation_pattern = re.compile(r'[!,.?\'" ]+')
    phrase_pattern = None
    grid_list = {}

    @staticmethod
    def get_text_score(text: str, word_dict: dict, phrase_dict: dict) -> int:
        """
        Given a twitter text, returning the corresponding sentiment score
        Args:
            text: twitter text
            word_dict: word sentiment score dictionary
            phrase_dict: phrase sentiment score dictionary

        Returns: sentiment score

        """
        score = 0
        text = text.lower().strip()
        # find the phrase first
        phrases = Utils.phrase_pattern.findall(text)
        if len(phrases) > 0:
            for phrase in phrases:
                score += phrase_dict[phrase]
            text = Utils.phrase_pattern.sub("", text)
        # find the word occurring in text
        for word in Utils.punctuation_pattern.split(text):
            if word in word_dict:
                score += word_dict[word]
        return score

    @staticmethod
    def get_grid(coordinates: tuple) -> str:
        """
        Given a coordinates returning the grid name where coordinates locates in
        Args:
            coordinates: x, y coordinates

        Returns: the name of grid name

        """
        x, y = coordinates
        for grid_name, grid_value in Utils.grid_list.items():
            if grid_value["xmin"] <= x <= grid_value["xmax"] and grid_value["ymin"] <= y <= grid_value["ymax"]:
                return grid_name
        return ""

    @staticmethod
    def process_single_twitter_data(line: str, grid_counter: dict, word_dict: dict, phrase_dict: dict):
        """
        Process a twitter data line then add a record into grid counter
        Args:
            line: raw twitter data string (in json format)
            grid_counter: dictionary that statistic the twitter data
            word_dict: word sentiment score dictionary
            phrase_dict: phrase sentiment score dictionary

        Returns: grid_counter

        """
        if line.startswith('{"id"'):
            coordinates = Utils.coordinates_pattern.search(line).groups()[0].split(",")
            x, y = coordinates
            coordinates = (float(x), float(y))
            text = Utils.text_pattern.search(line).groups()[0]
            grid = Utils.get_grid(coordinates)
            score = Utils.get_text_score(text, word_dict, phrase_dict)
            if grid in grid_counter:
                grid_counter[grid]["total"] += 1
                grid_counter[grid]["score"] += score
            return grid_counter

    @staticmethod
    def load_sentiment_score(file: str = "AFINN.txt"):
        """
        Load sentiment score file
        Args:
            file: filename of the sentiment score file, default is AFINN.txt

        Returns: word score dict and phrase score dict

        """
        word_score = {}
        phrase_score = {}
        with open(file, "r") as f:
            for line in f:
                tokens = line.split()
                if len(tokens) > 2:
                    phrase = " ".join(tokens[:-1])
                    phrase_score[phrase] = int(tokens[-1])
                else:
                    word, score = tokens
                    word_score[word] = int(score)
        pattern = "|".join(phrase_score.keys())
        Utils.phrase_pattern = re.compile(fr'\b({pattern})\b')
        return word_score, phrase_score

    @staticmethod
    def load_grid_file(file: str = "melbGrid.json"):
        """
        Load grid file and initialise grid counter and grid list
        Args:
            file: filename of the grid file, default is melbGrid.json

        Returns: grid counter, grid list

        """
        grid_counter = {}
        with open(file, "r") as f:
            grid_data = json.load(f)
        for grid_feature in grid_data["features"]:
            row = grid_feature["properties"]
            name = row["id"]
            grid = {"xmin": float(row["xmin"]), "xmax": float(row["xmax"]), "ymin": float(row["ymin"]),
                    "ymax": float(row["ymax"])}
            Utils.grid_list[name] = grid
            grid_counter[name] = {"total": 0, "score": 0}
        return grid_counter


def main():
    start = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_file", type=str, default='melbGrid.json', help='Path to the melbGrid json File')
    parser.add_argument('--twitter_file', type=str, default='smallTwitter.json', help='Path to the twitter data '
                                                                                      'json File')
    parser.add_argument("--score_file", type=str, default="AFINN.txt", help="Path to the sentiment score file")
    parser.add_argument("--batch_size", type=int, default=10, help='Number of data in batch for subprocesses to handle')
    opt = parser.parse_args()

    global GRID_FILE, TWITTER_FILE, SCORE_FILE
    GRID_FILE = opt.grid_file
    TWITTER_FILE = opt.twitter_file
    SCORE_FILE = opt.score_file
    batch_size = opt.batch_size

    # Initialise MPI necessary variable
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # load grid file and sentiment score file
    grid_counter = Utils.load_grid_file(GRID_FILE)
    word_score, phrase_score = Utils.load_sentiment_score(SCORE_FILE)

    if comm_size == 1:
        # one core
        with open(TWITTER_FILE, "r", encoding="utf8") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            while True:
                line = mm.readline().strip().decode("utf8")
                if not line:
                    break
                Utils.process_single_twitter_data(line, grid_counter, word_score, phrase_score)
            mm.close()
        end = datetime.datetime.now()
        print(f"total time consumes: {end - start}")
        print("Cell #Total Tweets #Overal Sentiment Score")
        for name, attr in grid_counter.items():
            print(f"{name} {attr['total']:13} {attr['score']:+23}")
    else:
        # multi core
        with open(TWITTER_FILE, "r", encoding="utf8") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            file_size = mm.size()
            block_size = file_size // comm_size
            # move to offset size
            mm.seek(comm_rank * block_size)
            block_end = (comm_rank + 1) * block_size
            index = mm.tell()
            while index <= block_end and index < file_size:
                line = mm.readline().strip().decode("utf8")
                index = mm.tell()
                try:
                    Utils.process_single_twitter_data(line, grid_counter, word_score, phrase_score)
                except Exception:
                    continue
            mm.close()

        # gather result
        combine_data = comm.gather(grid_counter, root=0)
        if comm_rank == 0:
            # merge result
            final_results = combine_results(combine_data)
            end = datetime.datetime.now()
            print(f"total time consumes: {end - start}")
            print("Cell #Total Tweets #Overal Sentiment Score")
            for name, attr in final_results.items():
                print(f"{name} {attr['total']:13} {attr['score']:+23}")


def combine_results(results: list) -> dict:
    """
    Combine and merge the sub-results from slave process into a final result
    Args:
        results: grid counter list

    Returns: combined grid counter

    """
    final_result = {}
    for result in results:
        if not result:
            continue
        for name, counter in result.items():
            if name not in final_result:
                final_result[name] = {"total": 0, "score": 0}
            final_result[name]["total"] += counter["total"]
            final_result[name]["score"] += counter["score"]
    return final_result


if __name__ == '__main__':
    main()
