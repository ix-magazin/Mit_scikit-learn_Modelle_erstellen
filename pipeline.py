# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3.10.9 ('challenge3.10')
#     language: python
#     name: python3
# ---

# # Code zum Artikel "Eine Einführung in scikit-learn"

# Dieses Notebook enthält einen (verkürzt dargestellten) Workflow zur Datenvorverarbeitung und Entwicklung eines Regressionsmodells zur Vorhersage von Gebrauchtwagenpreisen.

# +
import pandas as pd
import datetime
import plotly_express as px
from sklearn import set_config
from sklearn.compose import (
    make_column_transformer,
)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    mutual_info_regression,
)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
)
import shap

pd.options.plotting.backend = "plotly"

# -

# definition of some colors for plots
COLORS = [
    "#2f4463",
    "#04bfbf",
    "#89cff0",
    "#4d42bf"]

# ## Laden der Daten / Datenbereinigung
# *   Laden einer csv-Datei mit Daten zu gebrauchten Autos (Quelle: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
# *   Löschen der Datensatzspalten, die für das Problem irrelevant sind
# *   Auswahl eines Subdatensatzes
#     * nur die 15 meistverkauften Modelle
#     * ohne Outlier

# load data
data_file = "./data/vehicles.csv"
df = pd.read_csv(data_file)

# drop unwanted columns
drop_cols = [
    "id",
    "url",
    "region",
    "region_url",
    "image_url",
    "description",
    "county",
    "state",
    "VIN",
    "posting_date",
    "title_status",
    "lat",
    "long",
]
df = df.drop(columns=drop_cols, errors="ignore")


# exemplary data excerpt
df[30004:30009]

# +
# delete duplicated values
old_length = len(df)
df = df[~df.duplicated()]
new_length = len(df)

print(f"The number of rows was reduced from {old_length} to {new_length}")
# -

# take only x (20) most used models
num_models = 20
most_used_models = df["model"].value_counts()[:num_models].index.to_list()
df = df[df.model.isin(most_used_models)]
df = df.drop(columns=["model"])

px.histogram(df[df.price < 100000], x="price", title="Price histogram before cleaning prices")

# delete rather rare prices and too unrealistic prices
df = df[df.price < 50000]
df = df[df.price > 200]

px.histogram(df, x="price", title="Price histogram after cleaning prices")

# +
# delete unrealistic prices for unknown conditions?
df = df[~((df.price < 1000) & ~(df.condition.isna()))]

# unrealistic kilometers (too high) and too low (as cars should be used cars)
df = df[(df.odometer < 300000) & (df.odometer > 500)]
# -

px.histogram(df, x="odometer", title="Odometer histogram after cleaning odometer data")

print(f"The cleaned dataframe contains {len(df)} rows")

# ## Datenvorverarbeitung
# - One-Hot-Encoding der Kategoriespalten
# - Berechnung des Alters eines Autos

category_cols = [
    "manufacturer",
    "condition",
    "cylinders",
    "fuel",
    "transmission",
    "drive",
    "size",
    "type",
    "paint_color",
]

# split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["price"]), df["price"], test_size=0.2, random_state=42
)

# define data preprocessing steps and combine them into a ColumnTransformer
one_hot_encoder = OneHotEncoder(
    dtype=int, handle_unknown="infrequent_if_exist", sparse_output=False
)
age_calculator = FunctionTransformer(
    lambda x: datetime.date.today().year - x, feature_names_out="one-to-one"
)
column_transformer = make_column_transformer(
    (one_hot_encoder, category_cols),
    (age_calculator, ["year"]),
    ("passthrough", ["odometer"]),
    verbose_feature_names_out=False,
).set_output(transform="pandas")


# create train features using previously defined steps
train_features = column_transformer.fit_transform(X_train)
train_features.head()


# Die Trainingsdaten enthalten mehr Spalten als die Ausgangsdaten, da für jede Kategorie einer Kategoriespalte mit dem OneHotEncoding eine separate Spalte erstellt wurde

# ## Feature-Auswahl und Modelltraining
# - Auswahl der 15 besten Features, anhand der Metrik "mutual info regression"
# - Training der gesamten Pipeline: Datenvorverarbeitung, Featureauswahl, Training eines Regressionsalgorithmus

# define feature selector and full pipeline containing preprocessing and algorithm
feature_selector = SelectKBest(mutual_info_regression, k=15).set_output(
    transform="pandas"
)
pipe = Pipeline(
    [
        ("column_transformer", column_transformer),
        ("column_selection", feature_selector),
        ("model", GradientBoostingRegressor(random_state=2)),
    ]
)

# +
# Display Pipeline
set_config(display="diagram")

# fit data
pipe.fit(X_train, y_train)
# -

y_pred = pipe.predict(X_test)
print(f"Mean absolute error: {mean_absolute_error(y_test, y_pred)}")

# ### Kreuzvalidierung

cv_result = cross_validate(pipe, X_train, y_train, cv=10,
						   scoring="neg_mean_absolute_error", return_estimator=True)

cv_result["test_score"]

cv_result["test_score"].mean()

# Der Fehler ist hier etwas höher als bei der vorherigen Pipeline. Dies macht jedoch Sinn, da jeweils nicht auf den vollständigen Trainingsdaten trainiert wurde. Vorteilhaft bei der Kreuzvalidierung ist jedoch, dass das Risiko von Overfitting bei der Modellauswahl vermindert wird

# take the 8th model and get an idea of the error using the test set
y_pred_cv = cv_result["estimator"][7].predict(X_test)
mean_absolute_error(y_test, y_pred_cv)

# ### Interpretation der Modellergebnisse

# get feature names of features which got selected
feature_names = pipe[1].get_feature_names_out().tolist()
feature_names

# +
# plot feature importances
feature_importances = pipe[-1].feature_importances_
feature_series = pd.Series(dict(zip(feature_names, feature_importances)))

fig = px.bar(feature_series.sort_values(ascending=True), orientation='h', color_discrete_sequence=[COLORS[1]])
fig.layout.showlegend=False
fig.layout.width=600
fig.layout.yaxis.title = ""
fig.layout.xaxis.title = "Wichtigkeit Feature"
fig.layout.title = "Wichtigkeit von Features im Gradient Boosting-Modell"
fig.show()
# -

# calculate shap values using first trained pipeline
transformed_test_data = pipe[:-1].transform(X_test)
explainer = shap.TreeExplainer(pipe[-1], feature_names=transformed_test_data.columns.to_list())
shap_values = explainer.shap_values(transformed_test_data)

shap.summary_plot(shap_values, transformed_test_data, plot_type="bar")

fig = px.scatter(x=y_test, y=y_pred_cv, color_discrete_sequence=[COLORS[1]])
fig.layout.xaxis.title = "Wahrer Preis ($)"
fig.layout.yaxis.title = "Vorhergesagter Preis ($)"
fig.layout.width = 600
fig.layout.title = "Abweichung wahrer Preis und Preisprognose"
fig.show()

# +
# plot prognosis vs true values in a different way
import matplotlib.pyplot as plt

plt.hexbin(y_test, y_pred, gridsize=50, bins="log")
plt.colorbar()
plt.title("Abweichung wahrer Preis und Preisprognose")
plt.xlabel("Wahrer Preis ($)")
plt.ylabel("Preisprognose ($)")
# -

# ## Hyperparameter-Tuning

# +
# defining a hyperparameter search for the previously defined pipeline

search_params = {
    "model__loss": ["squared_error", "absolute_error"],
    "model__learning_rate": [0.5, 0.25, 0.1, 0.05, 0.01, 0.001],
    "model__n_estimators": [1, 2, 4, 8, 16, 32, 64, 100, 200],
    "model__max_depth": list(range(1, 10)),
    "model__min_samples_split": list(range(1, 10)),
    "model__min_samples_leaf": list(range(1, 10)),
    "model__max_features": list(range(2, X_train.shape[1])),
}
search = RandomizedSearchCV(
    pipe,
    search_params,
    n_jobs=4,
    cv=10,
    n_iter=20,
    verbose=1,
    scoring="neg_mean_absolute_error",
    random_state=2,
)
search.fit(X_train, y_train)
print("Best Score: ", search.best_score_)
print("Best Params: ", search.best_params_)
