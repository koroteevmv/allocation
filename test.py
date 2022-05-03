""" Тестовый файл. Содержит тестовые прогоны модели
"""
import logging
import pickle

import pandas as pd
from pulp import *

from Model import Model

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


def exec_small():
    """ Самая маленькая модель на основе предметов ПИ
    """
    model = Model()
    model.read_courses_hours(pd.read_csv("sample_data/small/courses.csv", index_col=0).fillna(0.0).sort_index())
    model.read_stuff_hours(pd.read_csv("sample_data/small/stuff.csv", index_col=0).fillna(0.0).sort_index())
    model.read_courses_tags(pd.read_csv("sample_data/small/courses_tags.csv", index_col=0).fillna(0.0).sort_index())
    model.read_stuff_tag(pd.read_csv("sample_data/small/stuff_tags.csv", index_col=0).fillna(0.0).sort_index())

    # model.calc_courses_tags()
    model.calc_method_stuff()
    #
    # model._solve_test()
    model.solve_gad()
    with open('./models/small', 'wb') as f:
        pickle.dump(model, f)


def exec_toy():
    """ Тестовая модель на основе предметов ПИ и ПМ
    """
    model = Model()
    model.read_courses_hours(pd.read_csv("sample_data/toy/courses.csv", index_col=0).fillna(0.0).sort_index())
    model.read_stuff_hours(pd.read_csv("sample_data/toy/stuff.csv", index_col=0).fillna(0.0).sort_index())
    model.read_courses_tags(pd.read_csv("sample_data/toy/courses_tags.csv", index_col=0).T.fillna(0.0).sort_index())
    model.read_stuff_tag(pd.read_csv("sample_data/toy/stuff_tags.csv", index_col=0).T.fillna(0.0).sort_index())

    model.calc_method_stuff()
    # model._solve_test()
    model.solve_gad()
    with open('./models/toy', 'wb') as f:
        pickle.dump(model, f)


def exec_2023():
    """ Модель нагрузки 2023 года. 1700 методических единиц, 130 преподавателей.
    """
    model = Model()
    model.read_courses_hours(pd.read_csv("sample_data/2023/courses.csv", index_col=0).fillna(0.0).sort_index())
    model.read_stuff_hours(pd.read_csv("sample_data/2023/stuff.csv", index_col=0).fillna(0.0).sort_index())
    model.read_stuff_tag(pd.read_csv("sample_data/2023/stuff_matrix.csv", index_col=0).fillna(0.0).sort_index())

    model.calc_courses_tags()
    model.calc_method_stuff()

    # model.solve_gad(generations=100, population=500)

    with open('./models/2023', 'wb') as f:
        pickle.dump(model, f)


def main():
    """ точка входа
    """
    exec_small()


if __name__ == '__main__':
    main()
