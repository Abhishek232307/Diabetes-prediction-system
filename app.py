from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from flask import Flask, render_template, request, send_file
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64


app = Flask(__name__)

# Load the dataset
df = pd.read_csv('diabetes.csv')
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, pd.NA)
df.fillna(df.mean(), inplace=True)

X = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train models
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


# Home Page
@app.route('/')
def home():
    return render_template('home.html')


# Form Page for each algorithm
@app.route('/form/<algo>')
def form(algo):
    algo_name = algo.replace('_', ' ').title()
    return render_template('form.html', algo=algo, algo_name=algo_name)


# Prediction Route
@app.route('/predict/<algo>', methods=['POST'])
def predict(algo):
    data = [float(request.form[key]) for key in
            ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']]
    new_data = pd.DataFrame([data], columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                                             'DiabetesPedigreeFunction', 'Age'])

    if algo == 'naive_bayes':
        model = nb_model
    elif algo == 'decision_tree':
        model = dt_model
    elif algo == 'random_forest':
        model = rf_model

    prediction = model.predict(new_data)[0]
    y_pred = model.predict(X_test)

    result = 'You might have diabetes.' if prediction == 1 else 'You do not have diabetes.'
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return render_template('result.html', result=result, accuracy=accuracy, f1_score=f1, precision=precision,
                           recall=recall)




# Visualization route
@app.route('/visualizations')
def visualizations():
    # Generate the visualizations
    # 1. Pie chart of diabetes outcome distribution
    plt.figure(figsize=(6, 6))
    df['Outcome'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'lightcoral'], labels=['No Diabetes', 'Diabetes'])
    plt.title('Diabetes Outcome Distribution')
    pie_chart = io.BytesIO()
    plt.savefig(pie_chart, format='png')
    pie_chart.seek(0)
    pie_chart_base64 = base64.b64encode(pie_chart.getvalue()).decode('utf-8')
    plt.close()

    # 2. Bar plot for average glucose levels based on diabetes outcome
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Outcome', y='Glucose', data=df, palette='viridis')
    plt.title('Average Glucose Levels by Outcome')
    plt.xlabel('Outcome')
    plt.ylabel('Average Glucose Level')
    bar_chart = io.BytesIO()
    plt.savefig(bar_chart, format='png')
    bar_chart.seek(0)
    bar_chart_base64 = base64.b64encode(bar_chart.getvalue()).decode('utf-8')
    plt.close()

    # 3. Heatmap for correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    heatmap = io.BytesIO()
    plt.savefig(heatmap, format='png')
    heatmap.seek(0)
    heatmap_base64 = base64.b64encode(heatmap.getvalue()).decode('utf-8')
    plt.close()

    return render_template('visualizations.html', pie_chart=pie_chart_base64, bar_chart=bar_chart_base64, heatmap=heatmap_base64)
if __name__ == '__main__':
    app.run(debug=True)
