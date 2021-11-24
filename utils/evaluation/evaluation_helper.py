import os
import json
from utils.logger import logger


class ScoreManager:

    def __init__(self, result_path, name):
        if not os.path.isdir(result_path):
            logger.info('[CHECKPOINT] Creating model file:%s' % result_path)
            os.makedirs(result_path)
        self.result_path_human = os.path.join(result_path, name + '.txt')
        self.result_path = os.path.join(result_path, name + '.json')
        if os.path.exists(self.result_path) and os.path.exists(self.result_path_human):
            logger.info('[SCORE MANAGER] Find the score file.')
            self.result = json.load(open(self.result_path, 'r+', encoding='utf-8'))
            self.result_human = json.load(open(self.result_path_human, 'r+', encoding='utf-8'))
        else:
            logger.info('[SCORE MANAGER] Will use a new score file.')
            self.result = dict()
            self.result_human = dict()

    def update(self, key, value):
        self.result[key] = value
        self.result_human[key] = value
        json.dump(self.result, open(self.result_path, 'w+', encoding='utf-8'))
        json.dump(self.result_human, open(self.result_path_human, 'w+', encoding='utf-8'), indent=1)

    def update_group(self, group, key_values):
        if group not in self.result:
            self.result[group] = key_values
        else:
            for key in key_values.keys():
                self.result[group][key] = key_values[key]
        human_key_values = {}
        scores_suffix = '_Scores'
        scores_suffix_len = len(scores_suffix)
        for key in self.result[group]:
            if key[-scores_suffix_len:] != scores_suffix:
                human_key_values[key] = self.result[group][key]
        self.result_human[group] = human_key_values
        json.dump(self.result, open(self.result_path, 'w+', encoding='utf-8'))
        json.dump(self.result_human, open(self.result_path_human, 'w+', encoding='utf-8'), indent=1)
