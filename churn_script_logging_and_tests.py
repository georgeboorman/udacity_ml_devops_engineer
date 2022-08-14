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
	from churn_library import import_data, perform_eda
	df = import_data("data/bank_data.csv")
	perform_eda(df)
	image_path = "images/eda/"
	files = ["churn_histogram.png", "correlation_heatmap.png",
	"customer_age_histogram.png", "marital_status_bar_plot.png", 
	"Total_Trans_Ct_kde_plot.png"]
	actual_files = set(os.listdir(image_path))
	try:
		assert actual_files.issuperset(set(files))
		logging.info(f'Files were successfully saved in {image_path}')
	except AssertionError:
		logging.error(f'Not all files were successfully saved in {image_path}')
	try:
		assert [os.stat(str(image_path + file)).st_size > 0 for file in files]
	except FileNotFoundError:
		logging.error("One or more of the files are empty")

def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''
	from churn_library import import_data
	df = import_data(df)
	try:
		cat_columns = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
		logging.info("Categories: {}".format(cat_columns)
        quant_columns = df.columns.symmetric_difference(cat_cols)
        logging.info("Quant columns: {}".format(quant_columns))
	except:
		


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








