import numpy as np
import pandas as pd

class NetflixLoader:
    def __init__(self):
        self.df = pd.DataFrame.from_dict({'movie_id': [], 'avg_rating': []})
        self.ratings = []
    def load_file(self, filename):
        data = (open(filename, 'r+')).readlines()
        curr_movie_id = None
        curr_ratings = []
        info = {'movie_id': [], 'avg_rating': []}
        self.ratings = list(self.ratings)
        num_categories = 5
        for line in data:
            tokens = line.split(',')
            if len(tokens) == 1:
                if curr_movie_id is not None:
                    info['movie_id'].append(curr_movie_id)
                    info['avg_rating'].append(np.mean(curr_ratings))
                    ratings = np.zeros((num_categories))
                    for r in curr_ratings:
                        ratings[int(r)-1] += 1
                    self.ratings.append(ratings)
                curr_movie_id = int(line.split(':')[0])
                curr_ratings = []
            else:
                curr_ratings.append(float(tokens[1]))
        new_df = pd.DataFrame.from_dict(info)
        self.df = pd.concat((self.df, new_df))
        self.ratings = np.array(self.ratings)
