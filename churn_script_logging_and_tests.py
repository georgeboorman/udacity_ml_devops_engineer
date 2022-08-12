import os
import logging
import churn_library_solution as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):
	'''
	test perform eda function
	'''
	try:
        assert isinstance(df, pd.DataFrame)
        logging.info("Successfully previewed DataFrame")
        logging.info("{} has {} rows and {} columns".format(df.shape[0], df.shape[1]))
        logging.info("{}".format(df.isnull().sum()))
        logging.info("{}".format(df.describe()))
        return df.head(), "\n", df.shape, \
                "\n", df.isnull().sum(), \
                "\n", df.describe()
    except AttributeError:
        logging.error("{} must be a pandas DataFrame to perform EDA.".format(df))
        return "{} must be a pandas DataFrame to use the perform_eda function.".format(df)


def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''
	try:
        cat_columns = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
        logging.info("Categories: {}".format(cat_columns)
        quant_columns = df.columns.symmetric_difference(cat_cols)
        logging.info("Quant columns: {}".format(quant_columns))
        return "Categories: {}".format(cat_columns), \
                "\n", "Quant columns: {}".format(quant_columns)
	else:
		


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	pass








