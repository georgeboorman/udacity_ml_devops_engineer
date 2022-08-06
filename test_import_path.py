from churn_library import import_data
import logging

logging.basicConfig(
    filename='/logs/results.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s')

if __name__ == "__main__":
    import_data("/data/bank_data.csv")
    import_data([1,2,3])
    import_data("README.md")