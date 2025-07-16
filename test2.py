import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Charger les données
train = pd.read_csv("train.csv")

# 2. Créer la feature FamilySize
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

# 3. Extraire le titre depuis le nom
def get_title(name):
    return name.split(",")[1].split(".")[0].strip()

train["Title"] = train["Name"].apply(get_title)

# 4. Regrouper les titres rares
rare_titles = ['Lady','Monsieur','Madame']
train["Title"] = train["Title"].replace(rare_titles, "Rare")

# 5. Remplir les valeurs manquantes
train["Age"].fillna(train["Age"].median(), inplace=True)
train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)

# 6. Supprimer la colonne Name (inutile)
train = train.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"])

# 7. Transformer les variables catégorielles en numériques (one-hot)
train = pd.get_dummies(train, columns=["Sex", "Embarked", "Title"], drop_first=True)

# 8. Séparer les données en features X et cible y
X = train.drop("Survived", axis=1)
y = train["Survived"]

# 9. Diviser en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. Créer et entraîner le modèle RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 11. Faire des prédictions sur le test
y_pred = model.predict(X_test)

# 12. Afficher les résultats
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))