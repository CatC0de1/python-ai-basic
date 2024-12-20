from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = [
    ["muda", "overweight", "pria", 1, "Ali"],
    ["muda", "underweight", "pria", 0, "Edi"],
    ["tua", "average", "wanita", 0, "Annie"],
    ["tua", "overweight", "pria", 0, "Budiman"],
    ["tua", "overweight", "pria", 1, "Herman"],
    ["tua", "underweight", "pria", 0, "Didi"],
    ["tua", "overweight", "wanita", 1, "Rina"],
    ["tua", "average", "pria", 0, "Gatot"],
]

map_usia = {"muda": 0, "tua": 1}
map_berat = {"underweight": 0, "average": 1, "overweight": 2}
map_kelamin = {"pria": 0, "wanita": 1}

X = [[map_usia[row[0]], map_berat[row[1]], map_kelamin[row[2]]] for row in data]
y = [row[3] for row in data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy * 100, "%")
print("\nDecision Tree Rules:")
tree_rules = export_text(model, feature_names=["Usia", "Berat Badan", "Jenis Kelamin"])
print(tree_rules)