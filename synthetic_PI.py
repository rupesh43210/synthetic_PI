import pandas as pd 
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import uuid
import multiprocessing
from tqdm import tqdm
import os
import psutil
import gc
from pathlib import Path
import logging
import json
import warnings
warnings.filterwarnings('ignore')

class MassivePIGenerator:
    def __init__(self, 
                 num_records: int = 100_000_000, 
                 chunk_size: int = 500_000, 
                 num_processes: int = None,
                 seed: int = 42):
        """Initialize the PI data generator"""
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        self.num_records = num_records
        self.chunk_size = chunk_size
        self.num_processes = num_processes or max(1, multiprocessing.cpu_count() - 1)
        
        self.setup_logging()
        self.load_reference_data()
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pi_generator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_reference_data(self):
        """Load reference data for consistent value generation"""
        self.reference_data = {
            'conditions': [
                'Diabetes', 'Hypertension', 'Asthma', 'Arthritis', 'Depression',
                'Anxiety', 'COPD', 'Cancer', 'Heart Disease', 'Allergies'
            ],
            'medications': [
                'Metformin', 'Lisinopril', 'Albuterol', 'Ibuprofen', 'Sertraline',
                'Alprazolam', 'Omeprazole', 'Levothyroxine', 'Atorvastatin', 'Amlodipine'
            ],
            'blood_types': ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-'],
            'insurance_types': ['PPO', 'HMO', 'EPO', 'POS', 'HDHP'],
            'departments': ['IT', 'HR', 'Finance', 'Sales', 'Marketing', 'Operations', 'Legal'],
            'education_levels': ['High School', 'Bachelor', 'Master', 'PhD', 'Associate'],
            'marital_status': ['Single', 'Married', 'Divorced', 'Widowed'],
            'ethnicities': ['Caucasian', 'African American', 'Asian', 'Hispanic', 'Native American']
        }

    def generate_chunk(self, chunk_id: int) -> dict:
        """Generate a chunk of PI data"""
        size = min(self.chunk_size, self.num_records - (chunk_id * self.chunk_size))
        if size <= 0:
            return None

        data = {
            # Identifiers
            'id': [str(uuid.uuid4()) for _ in range(size)],
            'ssn': [f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}" 
                   for _ in range(size)],
            
            # Personal Info
            'full_name': [self.fake.name() for _ in range(size)],
            'dob': [self.fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d') 
                   for _ in range(size)],
            'gender': np.random.choice(['M', 'F', 'NB'], size=size),
            'ethnicity': np.random.choice(self.reference_data['ethnicities'], size=size),
            'marital_status': np.random.choice(self.reference_data['marital_status'], size=size),
            
            # Contact Info
            'email': [self.fake.email() for _ in range(size)],
            'phone': [f"{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}" 
                     for _ in range(size)],
            'address': [self.fake.street_address() for _ in range(size)],
            'city': [self.fake.city() for _ in range(size)],
            'state': [self.fake.state() for _ in range(size)],
            'zip_code': [self.fake.zipcode() for _ in range(size)],
            
            # Financial Info
            'credit_card': [f"{random.randint(1000,9999)}-{random.randint(1000,9999)}-"
                          f"{random.randint(1000,9999)}-{random.randint(1000,9999)}" 
                          for _ in range(size)],
            'bank_account': [str(random.randint(10000000, 99999999)) for _ in range(size)],
            'salary': np.random.uniform(30000, 200000, size=size).round(2),
            
            # Employment Info
            'employer': [self.fake.company() for _ in range(size)],
            'department': np.random.choice(self.reference_data['departments'], size=size),
            'job_title': [self.fake.job() for _ in range(size)],
            'hire_date': [self.fake.date_between(start_date='-10y').strftime('%Y-%m-%d') 
                         for _ in range(size)],
            
            # Medical Info
            'blood_type': np.random.choice(self.reference_data['blood_types'], size=size),
            'conditions': [','.join(np.random.choice(self.reference_data['conditions'], 
                                                   size=random.randint(0, 3), replace=False))
                         for _ in range(size)],
            'medications': [','.join(np.random.choice(self.reference_data['medications'], 
                                                    size=random.randint(0, 3), replace=False))
                         for _ in range(size)],
            
            # Insurance Info
            'insurance_provider': [self.fake.company() for _ in range(size)],
            'insurance_type': np.random.choice(self.reference_data['insurance_types'], size=size),
            'policy_number': [f"POL{random.randint(1000000, 9999999)}" for _ in range(size)]
        }
        
        return data

    def save_chunk(self, chunk_data: dict, output_file: str):
        """Save a chunk of data to parquet format"""
        if chunk_data:
            df = pd.DataFrame(chunk_data)
            df.to_parquet(output_file, index=False, compression='snappy')

    def save_to_excel(self, df: pd.DataFrame, base_path: str):
        """Save dataframe to Excel files (splits if needed)"""
        excel_row_limit = 1_000_000  # Slightly under Excel's limit
        total_rows = len(df)
        
        if total_rows > excel_row_limit:
            num_files = (total_rows + excel_row_limit - 1) // excel_row_limit
            self.logger.info(f"Splitting into {num_files} Excel files due to row limit...")
            
            for i in range(num_files):
                start_idx = i * excel_row_limit
                end_idx = min((i + 1) * excel_row_limit, total_rows)
                
                file_path = f"{base_path}_part_{i+1}.xlsx"
                self.logger.info(f"Saving Excel part {i+1} of {num_files} to {file_path}")
                
                chunk_df = df.iloc[start_idx:end_idx]
                chunk_df.to_excel(
                    file_path,
                    index=False,
                    engine='xlsxwriter'
                )
        else:
            file_path = f"{base_path}.xlsx"
            self.logger.info(f"Saving to single Excel file: {file_path}")
            df.to_excel(file_path, index=False, engine='xlsxwriter')

    def save_to_csv(self, df: pd.DataFrame, base_path: str, rows_per_file: int = 5_000_000):
        """Save dataframe to CSV files (splits into manageable chunks)"""
        total_rows = len(df)
        num_files = (total_rows + rows_per_file - 1) // rows_per_file
        
        if num_files > 1:
            self.logger.info(f"Splitting into {num_files} CSV files for manageability...")
            
            for i in range(num_files):
                start_idx = i * rows_per_file
                end_idx = min((i + 1) * rows_per_file, total_rows)
                
                file_path = f"{base_path}_part_{i+1}.csv"
                self.logger.info(f"Saving CSV part {i+1} of {num_files} to {file_path}")
                
                chunk_df = df.iloc[start_idx:end_idx]
                chunk_df.to_csv(file_path, index=False)
        else:
            file_path = f"{base_path}.csv"
            self.logger.info(f"Saving to single CSV file: {file_path}")
            df.to_csv(file_path, index=False)

    def generate_and_save(self, output_dir: str = 'pi_data', formats: list = ['parquet', 'csv', 'excel']):
        """Generate and save the complete dataset in multiple formats"""
        Path(output_dir).mkdir(exist_ok=True)
        start_time = datetime.now()
        
        # Calculate number of chunks
        num_chunks = (self.num_records + self.chunk_size - 1) // self.chunk_size
        
        self.logger.info(f"Starting data generation at: {start_time}")
        self.logger.info(f"Generating {self.num_records:,} records in {num_chunks} chunks...")
        self.logger.info(f"Using {self.num_processes} processes")
        
        # Process chunks and save to parquet
        chunk_files = []
        with multiprocessing.Pool(self.num_processes) as pool:
            for chunk_id in tqdm(range(num_chunks), desc="Generating chunks"):
                chunk_file = f"{output_dir}/chunk_{chunk_id}.parquet"
                chunk_files.append(chunk_file)
                
                chunk_data = self.generate_chunk(chunk_id)
                if chunk_data:
                    self.save_chunk(chunk_data, chunk_file)
                
                if chunk_id % 10 == 0:
                    gc.collect()
        
        # Read chunks and combine
        self.logger.info("Combining chunks...")
        dfs = []
        for chunk_file in tqdm(chunk_files, desc="Reading chunks"):
            df = pd.read_parquet(chunk_file)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Save in each requested format
        base_path = f"{output_dir}/pi_data_{self.num_records}"
        
        if 'parquet' in formats:
            self.logger.info("Saving to parquet format...")
            combined_df.to_parquet(f"{base_path}.parquet", index=False)
        
        if 'csv' in formats:
            self.logger.info("Saving to CSV format...")
            self.save_to_csv(combined_df, base_path)
        
        if 'excel' in formats:
            self.logger.info("Saving to Excel format...")
            self.save_to_excel(combined_df, base_path)
        
        # Clean up chunk files
        for chunk_file in chunk_files:
            os.remove(chunk_file)
        
        # Save metadata
        metadata = {
            'num_records': self.num_records,
            'formats_generated': formats,
            'date_generated': datetime.now().isoformat(),
            'generation_time': str(datetime.now() - start_time),
            'column_count': len(combined_df.columns),
            'memory_used_gb': psutil.Process().memory_info().rss / 1024 / 1024 / 1024
        }
        
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Data generation complete! Time taken: {datetime.now() - start_time}")
        self.logger.info(f"Output directory: {output_dir}")

def main():
    # Initialize generator with 100M records
    generator = MassivePIGenerator(
        num_records=100_000_000,  # 100 million records
        chunk_size=500_000,       # 500K records per chunk
        seed=42
    )
    
    # Generate and save data in all formats
    generator.generate_and_save(
        output_dir='pi_data_100m',
        formats=['parquet', 'csv', 'excel']
    )

if __name__ == "__main__":
    main()
