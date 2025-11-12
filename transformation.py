import numpy as np
import warnings
warnings.filterwarnings('ignore')

def log_transformation(X_train_num, X_test_num, logger):
    try:
        for col in X_train_num.columns:
            X_train_num[col + '_log'] = np.log(X_train_num[col] + 1)
            X_test_num[col + '_log'] = np.log(X_test_num[col] + 1)
        logger.info("✅ Log Transformation completed successfully.")
        return X_train_num, X_test_num
    except Exception as e:
        import sys
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f"Issue in log_transformation() at line {er_lin.tb_lineno} : due to {er_msg}")

def handle_outliers(X_train_num, X_test_num, logger):
    try:
        for col in X_train_num.columns:
            Q1 = X_train_num[col].quantile(0.25)
            Q3 = X_train_num[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            X_train_num[col] = np.clip(X_train_num[col], lower, upper)
            X_test_num[col] = np.clip(X_test_num[col], lower, upper)
        logger.info("✅ Outlier handling completed successfully using IQR method.")
        return X_train_num, X_test_num
    except Exception as e:
        import sys
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f"Issue in handle_outliers() at line {er_lin.tb_lineno} : due to {er_msg}")