import sklearn.metrics
import pandas as pd
import tpot
from tpot import TPOTClassifier

import time


class MultiTrain:
    def __init__(self):
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.dataset = None
        self.dataset_path = None

    def set_dataset(self, dataset_path, read_cached=False, read_fraction=1, train_cols_to_drop=None, target_col=''):
        if train_cols_to_drop is None:
            train_cols_to_drop = []
        self.dataset_path = dataset_path
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(dataset_path, index_col=[0])

        if not read_cached:
            train = self.dataset.sample(frac=read_fraction)
            X = train.drop(train_cols_to_drop, axis=1)
            y = train[target_col]
            self.X_train, self.X_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(X, y,
                                                                                                            stratify=y,
                                                                                                            test_size=0.33)

    def build_and_fit_classifier(self, dataset_name="", generations=[100], population_sizes=[100], offspring_sizes=[100],
                                 mutation_rates=[0.9], crossover_rates=[0.1]):

        time_tracker = list()

        for value in generations:
            print("-> Fitting - Generations:", value)
            classifier = tpot.TPOTClassifier(generations=value, max_time_mins=10, n_jobs=-1, verbosity=2, early_stop=4)
            start = time.time()
            classifier.fit(self.X_train, self.y_train)
            end = time.time()
            elapsed_time = end - start
            time_tracker.append({'Feature': 'Generations', 'Value': value, 'Elapsed_Time': elapsed_time,
                                 'Train_Score': classifier.score(self.X_train, self.y_train),
                                 'Test_Score': classifier.score(self.X_test, self.y_test)})
            pd.DataFrame.from_dict(time_tracker).to_csv(f'./models/{dataset_name}/tpot/logs/log.csv')

            # save model
            classifier.export(f'./models/{dataset_name}/tpot/tpot_{dataset_name}_generations_{value}.py')
            print(f"Finished - saved to './models/{dataset_name}/tpot/tpot_{dataset_name}_generations_{value}.py'")

        for value in population_sizes:
            print("-> Fitting - Population Size:", value)
            classifier = tpot.TPOTClassifier(generations=1, population_size=value, max_time_mins=10, n_jobs=-1, verbosity=2, early_stop=4)
            start = time.time()
            classifier.fit(self.X_train, self.y_train)
            end = time.time()
            elapsed_time = end - start
            time_tracker.append({'Feature': 'Population_Size', 'Value': value, 'Elapsed_Time': elapsed_time,
                                 'Train_Score': classifier.score(self.X_train, self.y_train),
                                 'Test_Score': classifier.score(self.X_test, self.y_test)})
            pd.DataFrame.from_dict(time_tracker).to_csv(f'./models/{dataset_name}/tpot/logs/log.csv')

            # save model
            classifier.export(f'./models/{dataset_name}/tpot/tpot_{dataset_name}_population-size_{value}.py')
            print(f"Finished - saved to './models/{dataset_name}/tpot/tpot_{dataset_name}_population-size_{value}.py'")

        for value in offspring_sizes:
            print("-> Fitting - Offspring Size:", value)
            classifier = tpot.TPOTClassifier(generations=1, offspring_size=value, max_time_mins=10, n_jobs=-1, verbosity=2, early_stop=4)
            start = time.time()
            classifier.fit(self.X_train, self.y_train)
            end = time.time()
            elapsed_time = end - start
            time_tracker.append({'Feature': 'Offspring_Size', 'Value': value, 'Elapsed_Time': elapsed_time,
                                 'Train_Score': classifier.score(self.X_train, self.y_train),
                                 'Test_Score': classifier.score(self.X_test, self.y_test)})
            pd.DataFrame.from_dict(time_tracker).to_csv(f'./models/{dataset_name}/tpot/logs/log.csv')

            # save model
            classifier.export(f'./models/{dataset_name}/tpot/tpot_{dataset_name}_offspring-size_{value}.py')
            print(f"Finished - saved to './models/{dataset_name}/tpot/tpot_{dataset_name}_offspring-size_{value}.py'")

        for mut_rate, cross_rate in zip(mutation_rates, crossover_rates):
            print(f"-> Fitting - Mutation Rate: {mut_rate}, Crossover Rate: {cross_rate}")
            classifier = tpot.TPOTClassifier(generations=1, mutation_rate=mut_rate, crossover_rate=cross_rate,
                                             max_time_mins=10, n_jobs=-1, verbosity=2, early_stop=4)
            start = time.time()
            classifier.fit(self.X_train, self.y_train)
            end = time.time()
            elapsed_time = end - start
            time_tracker.append({'Feature': 'Mutation_Crossover_Rates', 'Value': f'{mut_rate},{cross_rate}',
                                 'Elapsed_Time': elapsed_time, 'Train_Score': classifier.score(self.X_train, self.y_train),
                                 'Test_Score': classifier.score(self.X_test, self.y_test)})
            pd.DataFrame.from_dict(time_tracker).to_csv(f'./models/{dataset_name}/tpot/logs/log.csv')

            # save model
            classifier.export(f'./models/{dataset_name}/tpot/tpot_{dataset_name}_mut-cross-rate_{mut_rate},{cross_rate}.py')
            print(f"Finished - saved to './models/{dataset_name}/tpot/tpot_{dataset_name}_mut-cross-rate_{mut_rate},{cross_rate}.py'")

        # for value in crossover_rates:
        #     print("-> Fitting - Crossover Rate:", value)
        #     classifier = tpot.TPOTClassifier(crossover_rate=value, max_time_mins=3, n_jobs=-1, verbosity=2, early_stop=2)
        #     classifier.fit(self.X_train, self.y_train)
        #
        #     # save model
        #     classifier.export(f'./models/smoke/tpot/tpot_smoke_crossover-rate_{value}.py')
        #     print(f"Finished - saved to './models/smoke/tpot/tpot_smoke_crossover-rate_{value}.py'")

        print('End')

