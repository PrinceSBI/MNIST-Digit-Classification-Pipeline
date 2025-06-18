import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

import mlflow

class mnist_class:
    def __init__(self):
        pass

    def mnist_data_extractor(self):
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784')
        x, y = mnist['data'], mnist['target']

        return x, y

    def image_plotter(self, x, index):
        # x, y = mnist['data'], mnist['target']

        some_digit = x[index]
        some_digit_image = some_digit.reshape(28, 28)  # let's reshape to plot it

        plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
                interpolation='nearest')
        plt.axis("off")
        plt.show()

    def train_test_split(self, x, y, test_percentage=0.2):
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_percentage, shuffle=True, random_state=42)
        
        # x_train, x_test = x[:60000], x[6000:70000]
        # y_train, y_test = y[:60000], y[6000:70000]

        # shuffle_index = np.random.permutation(60000)
        # x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

        # Creating a 2-detector
        y_train = y_train.astype(np.int8)
        y_test = y_test.astype(np.int8)
        # y_train_2 = (y_train == '2')
        # y_test_2 = (y_test == '2')

        return x_train, x_test, y_train, y_test
    
    def classification_report(self, model, x_test, y_test, model_name=True):
        from sklearn.metrics import classification_report

        # Predict on the test set
        y_pred = model.predict(x_test)

        def output_gen(y):
            ans = []
            for i in list(y):
                ans.append(list(i).index(max(list(i))))
            return np.array(ans)

        if model_name == 'NN':
            y_pred = output_gen(y_pred)
            report = classification_report(y_test, y_pred)
            print(report)
        else:
            report = classification_report(y_test, y_pred)
            print(report)

        report_dict = classification_report(y_test, y_pred, output_dict=True)

        return report_dict
    
    def ML_Flow(self, url, experiment_name, params, model, model_name, report_dict):
        mlflow.set_experiment(experiment_name)
        mlflow.set_tracking_uri(url)

        with mlflow.start_run(run_name=model_name):
            if params==False:
                params
            else:
                mlflow.log_params(params)
            mlflow.log_metrics({
                'accuracy': report_dict['accuracy'],
                'recall_class_0': report_dict['0']['recall'],
                'recall_class_1': report_dict['1']['recall'],
                'f1_score_macro': report_dict['macro avg']['f1-score']
            })
            mlflow.sklearn.log_model(model, model_name)  

            # # Log your TensorFlow model with MLflow
            # model_info = mlflow.tensorflow.log_model(model, name="tensorflow_model")

            # # Later, load your model for inference
            # loaded_model = mlflow.tensorflow.load_model(
            #     model_info.model_uri
            # )  # The 'model_uri' attribute is in the format 'models:/<model_id>'

    def ML_Flow_register(self, model_name, run_id):
        # model_name = 'XGB-Smote'
        # run_id=input('Please type RunID')
        model_uri = f'runs:/{run_id}/model_name'

        with mlflow.start_run(run_id=run_id):
            mlflow.register_model(model_uri=model_uri, name=model_name)
        
        