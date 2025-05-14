import numpy as np 
import bentoml 
from bentoml.io import NumpyNdarray
from bentoml import Service

iris_clf_runner = bentoml.sklearn.get("iris_model:latest").to_runner()

svc = Service("iris_classifier", runners = [iris_clf_runner])

@svc.api(input=NumpyNdarray(),output=NumpyNdarray())
def classify(input_series:np.ndarray) -> NumpyNdarray:
    result = iris_clf_runner.predict.run(input_series)
    return result