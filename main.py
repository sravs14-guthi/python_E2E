import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import warnings
import sys
import joblib, json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from log_code import setup_logging
from transformation import log_transformation, handle_outliers

warnings.filterwarnings('ignore')
logger = setup_logging('main')


class Telco_Churn:
    def __init__(self, path, logger):
        """Load and initialize data"""
        self.logger = logger
        self.logger.info("Loading data...")
        self.df = pd.read_csv(path)
        self.logger.info(f"✅ Data loaded successfully : {self.df.shape}")

        # Drop customerID if present
        if 'customerID' in self.df.columns:
            self.df.drop(columns=['customerID'], inplace=True)

        # Separate features and target
        self.X = self.df.drop(columns=['Churn'])
        self.y = self.df['Churn']

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.logger.info(f"✅ Train/Test Split done. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")


    def missing_values(self):
        #Handle missing values using median/mode imputation
        try:
            self.logger.info("Handling missing values...")
            self.df = self.df.fillna(self.df.median(numeric_only=True))
            self.df = self.df.fillna(self.df.mode().iloc[0])
            self.logger.info("✅ Missing Value Handling Completed Successfully.")
            self.logger.info(f"Missing Values After Handling:\n{self.df.isnull().sum()}")
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            self.logger.error(f"Issue at line {er_lin.tb_lineno} : {er_msg}")

    def variable_transformation(self):
        #Apply log transformation on numeric columns
        try:
            self.logger.info("Performing variable transformation...")
            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')

            self.X_train_num, self.X_test_num = log_transformation(
                self.X_train_num, self.X_test_num, self.logger
            )
            self.logger.info("✅ Variable transformation completed successfully.")
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            self.logger.error(f"Issue at line {er_lin.tb_lineno} : {er_msg}")


    def outlier_handling(self):
        #Handle outliers using IQR method
        try:
            self.logger.info("Handling outliers...")
            self.X_train_num, self.X_test_num = handle_outliers(
                self.X_train_num, self.X_test_num, self.logger
            )
            self.logger.info("✅ Outlier handling successfully completed.")
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            self.logger.error(f"Issue at line {er_lin.tb_lineno}: {er_msg}")


    def visualize_data(self):
        #Visualize key insights
        try:
            self.logger.info("Creating visualizations...")

            # Churn distribution
            plt.figure(figsize=(6, 4))
            self.df["Churn"].value_counts().plot(kind="bar", color=["skyblue", "salmon"])
            plt.title("Churn Distribution")
            plt.xlabel("Churn")
            plt.ylabel("Count")
            plt.show()

            # Monthly charges distribution
            plt.figure(figsize=(8, 5))
            sns.histplot(self.df["MonthlyCharges"], kde=True, color="skyblue")
            plt.title("Distribution of Monthly Charges")
            plt.show()

            # Tenure vs Churn
            plt.figure(figsize=(8, 5))
            sns.boxplot(x="Churn", y="tenure", data=self.df, palette="Set2")
            plt.title("Tenure vs Churn")
            plt.show()

            self.logger.info("✅ Visualization completed successfully.")
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            self.logger.error(f"Issue in visualize_data() at line {er_lin.tb_lineno}: {er_msg}")

    def preprocess_data(self):
        #Encode categorical variables and scale numeric features"""
        try:
            self.logger.info("Preprocessing data...")
            df = self.df.copy()

            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

            cat_cols = df.select_dtypes(include=["object"]).columns
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

            X = df.drop("Churn", axis=1)
            y = df["Churn"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=y
            )

            self.logger.info("✅ Encoding and scaling completed.")
            return X_train, X_test, y_train, y_test, scaler, X

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            self.logger.error(f"Issue in preprocess_data() at line {er_lin.tb_lineno}: {er_msg}")

    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and compare performance"""
        try:
            self.logger.info("Training models...")
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC(probability=True, random_state=42),
            }

            results = []
            plt.figure(figsize=(8, 6))

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob)

                fpr, tpr, _ = roc_curve(y_test, y_prob)
                plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

                results.append((name, acc, auc))

            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves for Models")
            plt.legend()
            plt.show()

            result_df = pd.DataFrame(results, columns=["Model", "Accuracy", "AUC"])
            print("\nModel Comparison:\n", result_df)

            best_model = result_df.loc[result_df["AUC"].idxmax()]
            self.logger.info(f"✅ Best model: {best_model['Model']} (AUC = {best_model['AUC']:.3f})")
            return best_model["Model"]

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            self.logger.error(f"Issue in train_models() at line {er_lin.tb_lineno}: {er_msg}")


    def find_best_parameters(self, X_train, y_train):
        """Tune RandomForest parameters"""
        try:
            self.logger.info("Performing hyperparameter tuning for RandomForest...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            }

            grid = GridSearchCV(
                estimator=RandomForestClassifier(random_state=42),
                param_grid=param_grid,
                scoring='roc_auc',
                cv=3,
                n_jobs=-1
            )

            grid.fit(X_train, y_train)
            self.logger.info(f"✅ Best Parameters: {grid.best_params_}")
            self.logger.info(f"✅ Best ROC-AUC: {grid.best_score_:.4f}")
            print(f"\nBest Parameters: {grid.best_params_}")
            print(f"Best ROC-AUC: {grid.best_score_:.4f}")

            return grid.best_estimator_

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            self.logger.error(f"Issue in find_best_parameters() at line {er_lin.tb_lineno}: {er_msg}")



# MAIN EXECUTION
if __name__ == "__main__":
    path ='C:\\Users\\user\\Downloads\\Telco_churn_project\\archive (1).zip'

    obj = Telco_Churn(path, logger)
    obj.missing_values()
    obj.variable_transformation()
    obj.outlier_handling()
    obj.visualize_data()

    X_train, X_test, y_train, y_test, scaler, X = obj.preprocess_data()
    best_model = obj.train_models(X_train, X_test, y_train, y_test)
    obj.find_best_parameters(X_train, y_train)

    # SAVE TRAINED MODEL AND SCALER
    try:
        logger.info("Saving final model and related files...")

        best_model_obj = RandomForestClassifier(
            random_state=42,
            max_depth=10,
            n_estimators=200,
            min_samples_split=10
        )
        best_model_obj.fit(X_train, y_train)

        # Save model
        joblib.dump(best_model_obj, "churn_model.pkl")
        logger.info("✅ Model saved as churn_model.pkl")

        # Save scaler
        joblib.dump(scaler, "scaler.pkl")
        logger.info("✅ Scaler saved as scaler.pkl")

        # Save feature list
        feature_names = list(X.columns)
        with open("features.json", "w") as f:
            json.dump(feature_names, f)
        logger.info("✅ Feature names saved as features.json")

        print("\n✅ Model, scaler, and feature list saved successfully!")

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.error(f"Issue while saving model at line {er_lin.tb_lineno}: {er_msg}")