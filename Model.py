"""Класс модели рассчета
"""
import logging

import numpy as np
import pandas as pd
import pygad
from pulp import makeDict, LpProblem, LpVariable, LpInteger, lpSum, PULP_CBC_CMD, value

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


class Model(object):
    """ Класс модели рассчета параметров нагрузки
    
    Свойства:
        stuff_hours:
            Матрица нагрузки преподавателей. Эта таблица загружается в модель на первом этапе.
            Колонки:
                индекс - имя преподавателя
                hours - часов нагрузки по плану. Это максимальное количество. Лучше его не превышать.

        method_hours:
            Расчетная матрица методических единиц. Она вычисляется из нагрузки в процессе загрузки. 
            Ее смысл - разделить часы в дисциплине на лекции, семинары. 
            При расчете этой таблицы используются настройки - кому какие часы записывать. 
            Это позволяет группировать нагрузку и не делить ее больше необходимого.
                
        list_courses: 
            Список дисциплин модели. Краткий перечень именно по дисциплинам. 
            Эта таблица вычисляется из списка нагрзуки по дисциплинам.
            
        list_stuff:
            Список преподавателей модели. Эта таблица вычисляется из поля stuff_hours и в дальнейшем
            держится в актуальном состоянии (например, при добавлении преподавателей из таблицы предпочтений).                
            
        courses_tags:
            Матрица соответствия дисциплин тегам. Эта матрица загружается в модель на втором этапе.
            По строкам - дисциплины, по колонкам - теги.
            
        stuff_tag:
            Таблица соответствия преподавателей тегам. Эта матрица загружается в модель на втором этапе.
            По строкам - преподаватели, по колонкам - теги.
            Список преподавателей должен соответствовать полям list_stuff и stuff_hours

        list_tags: 
            Список тегов (компетенций) модели. Эта таблица автоматически вычисляется из поля courses_tags

        method_tags:
            Соответствие методических единиц тегам модели. Эта таблица автоматически вычисляется из полей
            method_hours и courses_tags
            
        method_stuff:
            Соответствие преподавателей методическим единицам (итоговая матрица штрафов).
            Эта таблица автоматически вычисляется из полей
            method_tags и stuff_tag

        result: 
            В это поле загружается подробный отчет о распределении нагрузки после окончания моделирования.
            
        solution:
            В это поле сохраняется краткий вектор решения после окончания моделирования
    """
    def __init__(self):
        self.list_courses = None
        self.list_stuff = None
        self.list_tags = None

        self.method_hours = None
        self.stuff_hours = None

        self.courses_tags = None
        self.stuff_tag = None

        self.method_stuff = None
        self.method_tags = None

        self.result = None
        self.solution = None

        self.settings = {

        }

    def read_courses_hours(self, res):
        """ Чтение матрицы нагрузки

        :rtype:
            Model
        :type res:
            pd.DataFrame
        :param res:
            DataFrame, в колонках которого указаны дисциплины к распределению.
            По строкам может быть два варианта - либо одна колонка с объемом в часах, тогда дисциплина делится
            на методические единицы условно-фиксированного размера, либо 8 колонок с полной характеристикой
            нагрузки из ЕИС
        :return:
        объект модели
        """
        log.info("Матрица нагрузки прочтена")
        if self.list_courses is None:
            self.list_courses = res.index.drop_duplicates()
            log.info("Заполнен список дисциплин")
        else:
            if not self.list_courses.equals(res.index.drop_duplicates()):
                log.warning("Список дисциплин отличается:", self.list_courses, res.index.drop_duplicates())
                raise ValueError

        if len(res.columns) == 1:
            log.info("Работаем по упрощенной структуре нагрузки")
            res.columns = ['hours']
            res['Дисциплина'] = res.index.copy()
            res['Направление'] = 'undefined'
            res['Потоков'] = 3
            res['Групп'] = (res['hours'].astype('float64') / 72).astype('int64')
            res['Лекции'] = 0
            res['Cеминары'] = res['hours'].astype('float64')
            res['ПК'] = 0
            res['КР'] = 0
            res['ТК'] = 0
            res['Дисциплина'] = res.index.copy()
            res = res.drop(['hours'], axis=1)

        self.method_hours = self._expand_courses(res)
        log.debug(self.method_hours)
        return self

    @staticmethod
    def _expand_courses(res):
        """Разбор нагрузки на методические единицы
        """
        log.info("Работаем по полной структуре нагрузки")
        full_table = pd.DataFrame()
        for index, row in res.iterrows():
            if row["Лекции"] > 0 or row["ПК"] > 0:
                lections = pd.DataFrame([{
                    'course': index,
                    'type': 'lection',
                    'group': row["Направление"] + "_Поток_" + str(i + 1),
                    'hours': (row["Лекции"] + row["ПК"]) / row['Потоков']
                    # TODO предполагаем, что лектор проводит зачет или экзамен
                } for i in range(row['Потоков'])])
                full_table = pd.concat([full_table, lections], ignore_index=True)

            if row["Cеминары"] > 0 or row["ТК"] > 0:
                seminars = pd.DataFrame([{
                    'course': index,
                    'type': 'seminars',
                    'group': row["Направление"] + "_Группа_" + str(i + 1),
                    'hours': (row["Cеминары"] + row["ТК"]) / row['Групп']
                    # TODO предполагаем, что семинарист проводит текущий контроль
                } for i in range(row['Групп'])])
                full_table = pd.concat([full_table, seminars], ignore_index=True)

            if row["КР"] > 0:
                projects = pd.DataFrame([{
                    'course': index,
                    'type': 'projects',
                    'group': row["Направление"],
                    'hours': row["КР"]
                } for _ in range(1)])
                # TODO может, выделить индивидуальные курсовые работы?
                full_table = pd.concat([full_table, projects], ignore_index=True)

        return full_table

    def read_stuff_hours(self, res):
        """ Чтение матрицы нагрузки по преподавателям
        """
        res = res[~res.index.duplicated(keep='last')]
        if self.list_stuff is None:
            self.list_stuff = res.index
            log.info("Заполнен список преподавателей")
        else:
            if not self.list_stuff.equals(res.index):
                log.warning("Список преподавателей отличается:", self.list_stuff, res.index)
                raise ValueError

        if len(res.columns) == 1:
            res.columns = ['opt']
            res['opt'] = res['opt'].astype('float64')

        self.stuff_hours = res
        log.debug(self.stuff_hours)
        return self

    def read_courses_tags(self, res):
        """Чтение матрицы соответствия дисциплин и тегов
        """
        log.info("Прочтена матрица дисциплина-тег")
        if self.list_courses is None:
            self.list_courses = res.index
            log.info("Заполнен список дисциплин")
        else:
            if not self.list_courses.equals(res.index):
                log.warning(f"Список дисциплин отличается:\n{self.list_courses}\n{res.index}")
                raise ValueError
        if self.list_tags is None:
            self.list_tags = pd.Series(res.columns)
            log.info("Заполнен список тегов")
        else:
            if not self.list_tags.equals(pd.Series(res.columns)):
                log.warning("Список тегов отличается:", self.list_tags, pd.Series(res.columns))
                raise ValueError

        self.courses_tags = res
        log.debug(self.courses_tags)
        return self

    def read_stuff_tag(self, res):
        """ Чтение матрицы соответствия преподаватель-тег
        """
        log.info("Прочтена матрица преподаватель-тег")
        res = res[~res.index.duplicated(keep='last')]
        if self.list_stuff is None:
            self.list_stuff = res.index.drop_duplicates()
            log.info("Заполнен список преподавателей")
        else:
            if not self.list_stuff.equals(res.index.drop_duplicates()):
                log.warning("Список преподавателей отличается.")
                one, two = set(self.list_stuff), set(res.index.drop_duplicates())
                log.warning("В модели отсутствует информация о следующих преподавателях:")
                [log.warning(f"\t{x}") for x in sorted(two.difference(one))]
                log.warning("Их нагрузка не будет учтена.\n")
                log.warning("В предпочтениях отсутствует информация о следующих преподавателях:")
                [log.warning(f"\t{x}") for x in sorted(one.difference(two))]
                log.warning("Их предпочтения не будут учтены.")

                combined = sorted(one.union(two))
                self.stuff_hours = self.stuff_hours.reindex(combined).fillna(0)
                res = res.reindex(self.stuff_hours.index).fillna(0.5)
                self.list_stuff = self.stuff_hours.index

        if self.list_tags is None:
            self.list_tags = pd.Series(res.columns)
            log.info("Заполнен список тегов")
        else:
            if not self.list_tags.equals(pd.Series(res.columns)):
                log.warning("Список тегов отличается.")
                raise ValueError
            else:
                log.warning("Список тегов совпадает.")

        self.stuff_tag = res
        log.debug(self.stuff_tag)
        return self

    def calc_method_stuff(self):
        """ Вычисление матрицы соответствия методических единиц преподавателям (итоговая матрица штрафов).

        Матрица вычисляется из полей method_hours и courses_tags.
        """
        res = pd.DataFrame(index=self.method_hours.index, columns=self.list_tags)
        res = res.apply(lambda x: self.courses_tags.loc[self.method_hours.loc[x.name].course], axis=1)
        self.method_tags = res

        teacher_x_discipline = self.stuff_tag.dot(self.method_tags.T)
        penalty_matrix = teacher_x_discipline.T.iloc[:, :].div(res.T.sum(axis=0), axis=0)
        log.info("Вычислена итоговая матрица штрафов")
        self.method_stuff = penalty_matrix
        # penalty_matrix.index =
        log.debug(penalty_matrix)
        return penalty_matrix

    def calc_courses_tags(self):
        """Вычисление матрицы courses_tags из поля stuff_tag.

        Использование данного метода предполагает, что в поле stuff_tag в качестве тегов используются
        названия дисциплин
        """
        log.info("Запущен процесс генерации единичной матрицы дисциплин")
        tags = list(self.list_tags)
        courses = list(self.list_courses)
        if tags != courses:
            log.warning("Список дисциплин и тегов отличается!")
            one, two = set(tags), set(courses)
            log.warning("В тегах отсутствует информация о следующих дисциплинах:")
            [log.warning(f"\t{x}") for x in sorted(two.difference(one))]
            log.warning("В дисциплинах отсутствует информация о следующих тегах:")
            [log.warning(f"\t{x}") for x in sorted(one.difference(two))]

            log.debug(self.stuff_tag.shape)
            self.stuff_tag = (self.stuff_tag.T.reindex(self.list_courses).fillna(0.5)).T
            log.debug(self.stuff_tag.shape)
            self.list_tags = self.list_courses

        i = pd.DataFrame(np.identity(len(self.list_courses)),
                         columns=self.list_courses,
                         index=self.list_courses)
        log.debug(i)
        self.courses_tags = i

        return self

    def _solve_pulp(self):
        """ Поиск наилучшего распределения методом транспортной задачи.
        """
        log.info("Запущен процесс поиска оптимального решения")

        penalty_matrix = self.calc_method_stuff()

        warehouses = list(self.method_hours.index)
        log.debug(warehouses)

        supply = self.method_hours.hours.to_dict()  # TODO refactor
        log.debug(supply)

        bars = list(self.list_stuff)
        log.debug(bars)

        demand = self.stuff_hours.opt.to_dict()  # TODO refactor
        log.debug(demand)

        # TODO check the balance

        costs = penalty_matrix.to_numpy()
        log.debug(costs)

        costs = makeDict([warehouses, bars], costs, 0)
        prob = LpProblem("Stuff Allocation Problem",)
        routes = [(w, b) for w in warehouses for b in bars]
        vars_ = LpVariable.dicts("Route", (warehouses, bars), 0, None, LpInteger)
        prob += (
            lpSum([vars_[w][b] * costs[w][b] for (w, b) in routes]),
            "Sum_of_Penalties",
        )
        for r in routes:
            w, b = r
            prob += (
                lpSum([vars_[w][b]]) >= 0,
                "Always_Positive_%s_%s" % (w, b)
            )
        for w in warehouses:
            prob += (
                lpSum([vars_[w][b] for b in bars]) <= supply[w],
                "Sum_of_Products_out_of_Warehouse_%s" % w,
            )
        for b in bars:
            prob += (
                lpSum([vars_[w][b] for w in warehouses]) >= demand[b],
                "Sum_of_Products_into_Bar%s" % b,
            )
        prob.solve(PULP_CBC_CMD(msg=False))

        # The optimised objective function value is printed to the screen
        log.debug("Total Penalty = %.2f", value(prob.objective))
        result = pd.DataFrame(np.array([v.varValue for v in prob.variables()]).reshape((len(warehouses), -1)))
        result.index = self.method_hours.index
        result.columns = self.list_stuff
        log.info(result)
        log.info(result.sum().sum())

        self.result = result

        return self

    def _solve_test(self):
        """Генерация тестового решения модели. Необходимо для тестирования и воспроизводимости
        """
        log.debug(f"Количество методических единиц: {len(self.method_hours.index)}")
        log.debug(f"Количество преподавателей: {len(self.list_stuff)}")
        solution = [i % len(self.list_stuff) for i in range(len(self.method_hours.index))]
        self.result = self.evaluate_result(solution)
        self.solution = solution
        log.debug(self.result.sort_index())

    def solve_gad(self, generations=5, population=10):
        """ Поиск наилучшего распределения методом генетических алгоритмов

        :param generations: количество поколений генетического алгоритма
        :param population: количество особей в популяции
        """
        def _fitness(solution_, verbose=False):
            temp = log.getEffectiveLevel()
            log.setLevel(logging.DEBUG) if verbose else log.setLevel(logging.INFO)
            log.debug("Исследование пригодности решения")
            log.debug(solution_)

            series = self.evaluate_result(solution_)
            fitness_ = series["hours"] * series["fitness"]

            log.debug(fitness_)
            fitness_ = fitness_.sum() / series["hours"].sum()

            log.setLevel(temp)
            return fitness_

        def fitness(solution_, solution_idx):
            """Расчет общей эффективности распределения
            """
            return _fitness(solution_)

        def callback_gen(ga_instance_):
            """ Вывод промежуточной информации о ходе работы генетического алгоритма
            """
            print("Generation : ", ga_instance_.generations_completed)
            print("Fitness of the best solution :", ga_instance_.best_solution()[1])

        # def mutation_func(offspring, ga_instance_):
        #     for chromosome_idx in range(offspring.shape[0]):
        #         series = list(self.generate_result(offspring[chromosome_idx])["fitness"])
        #         fitness_sorted = sorted(list(range(len(series))), key=lambda k: series[k])
        #         random_gene_idx = np.random.choice(fitness_sorted[:25])
        #         offspring[chromosome_idx, random_gene_idx] = randint(0, len(self.list_stuff) - 1)
        #     return offspring

        log.debug(f"Количество методических единиц: {len(self.method_hours.index)}")
        log.debug(f"Количество преподавателей: {len(self.list_stuff)}")
        ga_instance = pygad.GA(num_generations=generations,
                               num_parents_mating=int(population/2),
                               sol_per_pop=population,

                               num_genes=len(self.method_hours.index),
                               init_range_low=0,
                               init_range_high=len(self.list_stuff),
                               random_mutation_min_val=0,
                               random_mutation_max_val=len(self.list_stuff),
                               gene_type=int,

                               fitness_func=fitness,
                               callback_generation=callback_gen,

                               mutation_by_replacement=True,
                               stop_criteria=["reach_1.0", "saturate_5"],
                               parent_selection_type="rank",
                               crossover_type="scattered",
                               # mutation_type=mutation_func,  # "random",
                               # mutation_probability=0.01,
                               )
        ga_instance.run()
        ga_instance.plot_fitness()
        solution = ga_instance.best_solution()[0]
        self.result = self.evaluate_result(solution)
        self.solution = solution

    def evaluate_result(self, solution):
        """ Расчет показателей распределения нагрузки по сгенерированному решению.

        :param solution:
        :return:
        """
        res = pd.DataFrame(index=range(len(self.method_hours.index)), columns=range(len(self.list_stuff))).fillna(0)
        for method, prep in enumerate(solution):
            res.loc[method, prep] = self.method_hours.hours.loc[method]
        res.index = self.method_hours.index
        res.columns = self.list_stuff

        series = pd.DataFrame(solution, columns=['stuff'])
        series.index = self.method_hours.index

        series["course"] = self.method_hours.course
        series["group"] = self.method_hours.group
        series["type"] = self.method_hours.type
        series["name"] = self.stuff_hours.index[series.stuff]
        series["hours"] = self.method_hours.hours

        delta = pd.DataFrame(res.sum(axis=0), columns=['real'])
        delta["opt"] = self.stuff_hours.opt
        series = pd.merge(series, delta, left_on='name', right_index=True)
        series["f_dist"] = series["real"] / series["opt"]
        series["f_dist"][series["f_dist"] <= 1] = 1
        series["f_dist"][series["f_dist"] > 1] = 0

        log.debug(series.index)
        log.debug(self.stuff_hours.index[series.stuff])
        series["f_skill"] = self.method_stuff.lookup(series.index, self.stuff_hours.index[series.stuff])

        series["fitness"] = (series["f_dist"] * series["f_skill"])
        return series
