from src.stockpredictor.utils.common import save_as_csv
from src.stockpredictor.logging.coustom_log import logger
from src.stockpredictor.entity import DataValidationConfig
import pandas as pd

class DataValidation:
    def __init__(self, config: DataValidationConfig) -> None:
        self.config = config

    
    def validate_data(self):
        try:
            # read the data from csv
            raw_data = pd.read_csv(self.config.validation_input,  parse_dates=['Date'])
            validation_status = True

            # check if no.of colums are correct
            validation_status = {}

            for column, expected_dtype in self.config.validation_schema.items():
                if column not in raw_data.columns:
                    validation_status[column] = False
                    continue  # Skip further checks for missing columns

                actual_dtype = raw_data[column].dtype
                if expected_dtype == 'datetime':
                    validation_status[column] = pd.api.types.is_datetime64_any_dtype(actual_dtype)
                elif expected_dtype == 'float':
                    validation_status[column] = pd.api.types.is_float_dtype(actual_dtype)
                elif expected_dtype == 'int':
                    validation_status[column] = pd.api.types.is_integer_dtype(actual_dtype)
                else:
                    validation_status[column] = False  

            # Check the overall validation status
            overall_status = all(validation_status.values())

            # Prepare the content to write to the file
            output_lines = []
            output_lines.append(f"Overall Validation Status: {'Success' if overall_status else 'Failure'}\n")
            output_lines.append("Column-wise Validation Status:\n")
            for col, status in validation_status.items():
                output_lines.append(f" - {col}: {'Passed' if status else 'Failed'}\n")

            # Write to the file file
            output_file = self.config.validation_report_path
            with open(output_file, "w") as f:
                f.writelines(output_lines)

            if overall_status:
                logger.info(f"Data validation successful. Report saved to {output_file}")
                save_as_csv(raw_data, self.config.validation_output_path)
            print(f"Validation status written to {output_file}")

        except Exception as e:
            raise e