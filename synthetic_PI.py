import pandas as pd 
import numpy as np
from faker import Faker
from datetime import datetime
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
warnings.filterwarnings('ignore')

class PIDataGenerator:
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.memory = psutil.virtual_memory()
        self.setup_logging()
        self.logger.info(f"System has {self.cpu_count} CPU cores")
        self.logger.info(f"Available memory: {self.memory.available / (1024**3):.2f} GB")
        
        # Dynamically calculate optimal chunk size based on available memory
        self.chunk_size = self.calculate_chunk_size()
        self.logger.info(f"Optimal chunk size: {self.chunk_size:,} records")

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pi_generator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def calculate_chunk_size(self):
        # Estimate memory per record (approximately 1KB per record)
        mem_per_record = 1024  # bytes
        available_mem = self.memory.available * 0.7  # Use 70% of available memory
        return min(500_000, int(available_mem / mem_per_record / self.cpu_count))

    @staticmethod
    def generate_chunk(args):
        chunk_id, chunk_size, seed = args
        
        # Initialize generators for this chunk
        fake = Faker()
        Faker.seed(seed + chunk_id)
        random.seed(seed + chunk_id)
        np.random.seed(seed + chunk_id)
        
        # Generate data using numpy where possible for better performance
        size = chunk_size
        
        # Generate arrays for repetitive data
        genders = np.random.choice(['M', 'F', 'NB'], size=size)
        salary_range = np.random.uniform(30000, 200000, size=size).round(2)
        blood_types = np.random.choice(['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-'], size=size)
        
        data = {
            'id': [str(uuid.uuid4()) for _ in range(size)],
            'ssn': [f"{random.randint(100,999):03d}-{random.randint(10,99):02d}-{random.randint(1000,9999):04d}" 
                   for _ in range(size)],
            'full_name': [fake.name() for _ in range(size)],
            'email': [fake.email() for _ in range(size)],
            'phone': [f"{random.randint(100,999):03d}-{random.randint(100,999):03d}-{random.randint(1000,9999):04d}" 
                     for _ in range(size)],
            'dob': [fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d') 
                   for _ in range(size)],
            'gender': genders,
            'address': [fake.street_address() for _ in range(size)],
            'city': [fake.city() for _ in range(size)],
            'state': [fake.state() for _ in range(size)],
            'zip_code': [fake.zipcode() for _ in range(size)],
            'credit_card': [f"{random.randint(1000,9999):04d}-{random.randint(1000,9999):04d}-"
                          f"{random.randint(1000,9999):04d}-{random.randint(1000,9999):04d}" 
                          for _ in range(size)],
            'bank_account': [f"ACCT{random.randint(10000000, 99999999):08d}" for _ in range(size)],
            'salary': salary_range,
            'employer': [fake.company() for _ in range(size)],
            'job_title': [fake.job() for _ in range(size)],
            'blood_type': blood_types,
            'insurance_id': [f"INS{random.randint(100000, 999999):06d}" for _ in range(size)]
        }
        
        # Save chunk to parquet
        df = pd.DataFrame(data)
        output_file = f"temp_chunks/chunk_{chunk_id}.parquet"
        df.to_parquet(output_file, index=False, compression='snappy')
        
        # Clean up
        del df
        del data
        gc.collect()
        
        return output_file

    def combine_chunks(self, chunk_files, output_prefix):
        """Combine chunks and save in multiple formats"""
        dfs = []
        for cf in chunk_files:
            df = pd.read_parquet(cf)
            dfs.append(df)
            os.remove(cf)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Save in multiple formats using threads
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Save parquet
            executor.submit(
                combined_df.to_parquet,
                f"output/{output_prefix}.parquet",
                index=False,
                compression='snappy'
            )
            
            # Save CSV
            executor.submit(
                combined_df.to_csv,
                f"output/{output_prefix}.csv",
                index=False
            )
            
            # Save Excel if size permits (under 1M rows)
            if len(combined_df) < 1_000_000:
                executor.submit(
                    combined_df.to_excel,
                    f"output/{output_prefix}.xlsx",
                    index=False,
                    engine='xlsxwriter'
                )
        
        del dfs
        del combined_df
        gc.collect()

    def generate_data(self, num_records=100_000_000, seed=42):
        """Main method to generate data using all available cores"""
        start_time = datetime.now()
        
        # Create directories
        os.makedirs('temp_chunks', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        
        # Calculate number of chunks
        num_chunks = (num_records + self.chunk_size - 1) // self.chunk_size
        
        self.logger.info(f"Starting data generation at: {start_time}")
        self.logger.info(f"Generating {num_records:,} records in {num_chunks} chunks")
        self.logger.info(f"Using all {self.cpu_count} CPU cores")
        
        # Prepare chunk arguments
        chunk_args = [(i, self.chunk_size, seed) for i in range(num_chunks)]
        
        # Generate chunks using all available cores
        chunk_files = []
        with ProcessPoolExecutor(max_workers=self.cpu_count) as executor:
            futures = list(tqdm(
                executor.map(self.generate_chunk, chunk_args),
                total=num_chunks,
                desc="Generating chunks"
            ))
            chunk_files.extend(futures)
            
            # Combine chunks periodically
            CHUNKS_PER_COMBINE = max(10, self.cpu_count)
            for i in range(0, len(chunk_files), CHUNKS_PER_COMBINE):
                batch = chunk_files[i:i + CHUNKS_PER_COMBINE]
                if batch:
                    self.combine_chunks(batch, f"combined_{i//CHUNKS_PER_COMBINE}")
        
        # Final metadata
        metadata = {
            'num_records': num_records,
            'num_chunks': num_chunks,
            'chunk_size': self.chunk_size,
            'cpu_cores_used': self.cpu_count,
            'date_generated': datetime.now().isoformat(),
            'generation_time': str(datetime.now() - start_time),
            'memory_used_gb': psutil.Process().memory_info().rss / 1024 / 1024 / 1024,
            'system_info': {
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'cpu_count': self.cpu_count,
                'platform': os.uname().sysname if hasattr(os, 'uname') else os.name
            }
        }
        
        with open('output/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Data generation complete! Time taken: {datetime.now() - start_time}")
        self.logger.info(f"Peak memory usage: {metadata['memory_used_gb']:.2f} GB")
        self.logger.info("Files saved in 'output' directory")

def main():
    generator = PIDataGenerator()
    generator.generate_data(num_records=100_000_000)

if __name__ == "__main__":
    main()
