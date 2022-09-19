import sklearn.metrics
import pandas as pd
import autosklearn.classification
from autosklearn.metrics import (accuracy, f1, roc_auc, precision,
                                 average_precision, recall, log_loss)
from sklearn.model_selection import StratifiedKFold
import pickle
import time


class MultiTrain:
    def __init__(self):
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.dataset = None
        self.dataset_path = None

    def set_dataset(self, dataset_path, read_fraction=1, train_cols_to_drop=None, target_col=''):
        if train_cols_to_drop is None:
            train_cols_to_drop = []
        self.dataset_path = dataset_path
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(dataset_path, index_col=[0])
        train = self.dataset.sample(frac=read_fraction)
        X = train.drop(train_cols_to_drop, axis=1)
        y = train[target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(X, y,
                                                                                                        stratify=y,
                                                                                                        test_size=0.33)

    def build_and_fit_classifier(self, dataset_name="", n_jobs=-1, time_left_for_this_task=180, per_run_time_limit=60,
                                 initial_configurations_via_metalearning=None,
                                 resampling_strategies=None,
                                 max_models_on_disc=None, ensemble_sizes=None, metric=None, scoring_functions=None):

        if initial_configurations_via_metalearning is None:
            initial_configurations_via_metalearning = [25]
        if resampling_strategies is None:
            resampling_strategies = [StratifiedKFold(n_splits=5)]
        if max_models_on_disc is None:
            max_models_on_disc = [50]
        if ensemble_sizes is None:
            ensemble_sizes = [50]

        time_tracker = list()

        for value in initial_configurations_via_metalearning:
            print("-> Fitting - Initial Configurations:", value)
            classifier = autosklearn.classification.AutoSklearnClassifier(
                n_jobs=n_jobs,
                time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit=per_run_time_limit,
                initial_configurations_via_metalearning=value
            )
            start = time.time()
            classifier.fit(self.X_train, self.y_train)
            end = time.time()
            elapsed_time = end - start
            classifier.refit(X=self.X_train, y=self.y_train)

            train_preds = classifier.predict(self.X_train)
            predictions = classifier.predict(self.X_test)

            train_acc_score = sklearn.metrics.accuracy_score(self.y_train, train_preds)
            test_acc_score = sklearn.metrics.accuracy_score(self.y_test, predictions)
            time_tracker.append({'Feature': 'Initial_Confs', 'Value': value, 'Elapsed_Time': elapsed_time,
                             'Train_Score': train_acc_score, 'Test_Score': test_acc_score})
            pd.DataFrame.from_dict(time_tracker).to_csv(f'./models/{dataset_name}/autosklearn/logs/log.csv')

            # save model
            with open(f'./models/{dataset_name}/autosklearn/autosklearn_{dataset_name}_initial-confs_{value}.pkl', 'wb') as f:
                pickle.dump(classifier, f)
            print(f"Finished - saved to './models/{dataset_name}/autosklearn/autosklearn_{dataset_name}_initial-confs_{value}.pkl'")

        for value in max_models_on_disc:
            print("-> Fitting - Max Models on Disc:", value)
            classifier = autosklearn.classification.AutoSklearnClassifier(
                n_jobs=n_jobs,
                time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit=per_run_time_limit,
                max_models_on_disc=value
            )
            start = time.time()
            classifier.fit(self.X_train, self.y_train)
            end = time.time()
            elapsed_time = end - start
            classifier.refit(X=self.X_train, y=self.y_train)

            train_preds = classifier.predict(self.X_train)
            predictions = classifier.predict(self.X_test)

            train_acc_score = sklearn.metrics.accuracy_score(self.y_train, train_preds)
            test_acc_score = sklearn.metrics.accuracy_score(self.y_test, predictions)
            time_tracker.append({'Feature': 'Max_Models', 'Value': value, 'Elapsed_Time': elapsed_time,
                                 'Train_Score': train_acc_score, 'Test_Score': test_acc_score})
            pd.DataFrame.from_dict(time_tracker).to_csv(f'./models/{dataset_name}/autosklearn/logs/log.csv')

            # save model
            with open(f'./models/{dataset_name}/autosklearn/autosklearn_{dataset_name}_max-models_{value}.pkl', 'wb') as f:
                pickle.dump(classifier, f)
            print(f"Finished - saved to './models/{dataset_name}/autosklearn/autosklearn_{dataset_name}_max-models_{value}.pkl'")

        for value in ensemble_sizes:
            print("-> Fitting - Ensemble Sizes:", value)
            classifier = autosklearn.classification.AutoSklearnClassifier(
                n_jobs=n_jobs,
                time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit=per_run_time_limit,
                ensemble_size=value
            )
            start = time.time()
            classifier.fit(self.X_train, self.y_train)
            end = time.time()
            elapsed_time = end - start
            classifier.refit(X=self.X_train, y=self.y_train)

            train_preds = classifier.predict(self.X_train)
            predictions = classifier.predict(self.X_test)

            train_acc_score = sklearn.metrics.accuracy_score(self.y_train, train_preds)
            test_acc_score = sklearn.metrics.accuracy_score(self.y_test, predictions)
            time_tracker.append({'Feature': 'Ensemble_Size', 'Value': value, 'Elapsed_Time': elapsed_time,
                                 'Train_Score': train_acc_score, 'Test_Score': test_acc_score})
            pd.DataFrame.from_dict(time_tracker).to_csv(f'./models/{dataset_name}/autosklearn/logs/log.csv')

            # save model
            with open(f'./models/{dataset_name}/autosklearn/autosklearn_{dataset_name}_ensemble-size_{value}.pkl', 'wb') as f:
                pickle.dump(classifier, f)
            print(f"Finished - saved to './models/{dataset_name}/autosklearn/autosklearn_{dataset_name}_ensemble-size_{value}.pkl'")

        for value in resampling_strategies:
            print("-> Fitting - Resampling Strategy:", value)
            classifier = autosklearn.classification.AutoSklearnClassifier(
                n_jobs=n_jobs,
                time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit=per_run_time_limit,
                resampling_strategy=value,
                resampling_strategy_arguments={'folds': 5}
            )
            start = time.time()
            classifier.fit(self.X_train, self.y_train)
            end = time.time()
            elapsed_time = end - start
            classifier.refit(X=self.X_train, y=self.y_train)

            train_preds = classifier.predict(self.X_train)
            predictions = classifier.predict(self.X_test)

            train_acc_score = sklearn.metrics.accuracy_score(self.y_train, train_preds)
            test_acc_score = sklearn.metrics.accuracy_score(self.y_test, predictions)
            time_tracker.append({'Feature': 'Resampling_Strategy', 'Value': value, 'Elapsed_Time': elapsed_time,
                                 'Train_Score': train_acc_score, 'Test_Score': test_acc_score})
            pd.DataFrame.from_dict(time_tracker).to_csv(f'./models/{dataset_name}/autosklearn/logs/log.csv')

            # save model
            with open(f'./models/{dataset_name}/autosklearn/autosklearn_{dataset_name}_resamp-strategy_{value}.pkl', 'wb') as f:
                pickle.dump(classifier, f)
            print(f"Finished - saved to './models/{dataset_name}/autosklearn/autosklearn_{dataset_name}_resamp-strategy_{value}.pkl'")


    def predict(self, model_name):
        with open(model_name, 'rb') as f:
            loaded_classifier = pickle.load(f)
        loaded_classifier.refit(X=self.X_train, y=self.y_train)
        predictions = loaded_classifier.predict(self.X_test)

        return sklearn.metrics.accuracy_score(self.y_test, predictions)
