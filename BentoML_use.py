import bentoml
from sklearn import datasets
from sklearn.svm import SVC

df = datasets.load_iris()

# split in input and output data
x = df.data 
y = df.target

# train the model
model = SVC(gamma='scale')
model.fit(x,y)

# save the model into BentoML local storage 
saved_model = bentoml.sklearn.save_model("iris_model" , model)
print(f"model train is succsefully: {saved_model}")

'''the tag is ("iris_model:wks42nrqrc6bopbb") '''