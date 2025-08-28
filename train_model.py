import joblib
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier

print("Iniciando o processo de treinamento...")

print("Carregando o dataset MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data / 255.0, mnist.target
print("Dataset carregado e normalizado.")

print("Treinando o SGDClassifier...")

y_train_5 = (y == '5')
sgd_clf = SGDClassifier(random_state=42, class_weight='balanced')
sgd_clf.fit(X, y_train_5)
print("Modelo treinado com sucesso.")

model_filename = 'sgd_model.joblib'
joblib.dump(sgd_clf, model_filename)

print(f"Modelo salvo com sucesso no arquivo: {model_filename}")
print("Processo de treinamento concluído.")

