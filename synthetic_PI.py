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
import signal
import atexit
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
warnings.filterwarnings('ignore')

class PIDataGenerator:
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.memory = psutil.virtual_memory()
        self.processes = []
        self.setup_logging()
        self.setup_cleanup_handlers()
        
        self.logger.info(f"System has {self.cpu_count} CPU cores")
        self.logger.info(f"Available memory: {self.memory.available / (1024**3):.2f} GB")
        
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

    def setup_cleanup_handlers(self):
        """Setup handlers for proper cleanup on exit"""
        def cleanup_handler(signum, frame):
            self.logger.info("Cleanup signal received. Terminating processes...")
            self.cleanup()
            exit(0)

        def exit_handler():
            self.cleanup()

        # Register signal handlers
        signal.signal(signal.SIGTERM, cleanup_handler)
        signal.signal(signal.SIGINT, cleanup_handler)
        atexit.register(exit_handler)

    def cleanup(self):
        """Clean up resources and temporary files"""
        self.logger.info("Starting cleanup...")
        
        # Kill any remaining child processes
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

        # Remove temporary files
        try:
            if os.path.exists('temp_chunks'):
                for file in os.listdir('temp_chunks'):
                    try:
                        os.remove(os.path.join('temp_chunks', file))
                    except Exception as e:
                        self.logger.error(f"Error removing temp file: {e}")
                os.rmdir('temp_chunks')
        except Exception as e:
            self.logger.error(f"Error cleaning temp directory: {e}")

        # Force garbage collection
        gc.collect()
        
        self.logger.info("Cleanup completed")

    def calculate_chunk_size(self):
        mem_per_record = 1024  # bytes
        available_mem = self.memory.available * 0.7  # Use 70% of available memory
        return min(500_000, int(available_mem / mem_per_record / self.cpu_count))

    @staticmethod
    def generate_chunk(args):
        chunk_id, chunk_size, seed = args
        
        # Register process for monitoring
        process = psutil.Process()
        process.nice(10)  # Lower priority to prevent system overload
        
        fake = Faker()
        Faker.seed(seed + chunk_id)
        random.seed(seed + chunk_id)
        np.random.seed(seed + chunk_id)
        
        size = chunk_size
        
        try:
            # Generate data
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
                'gender': np.random.choice(['M', 'F', 'NB'], size=size),
                'address': [fake.street_address() for _ in range(size)],
                'city': [fake.city() for _ in range(size)],
                'state': [fake.state() for _ in range(size)],
                'zip_code': [fake.zipcode() for _ in range(size)]
            }
            
            df = pd.DataFrame(data)
            output_file = f"temp_chunks/chunk_{chunk_id}.parquet"
            df.to_parquet(output_file, index=False, compression='snappy')
            
            # Cleanup
            del df
            del data
            gc.collect()
            
            return output_file
            
        except Exception as e:
            return f"ERROR: {str(e)}"

    def combine_chunks(self, chunk_files, output_prefix):
        """Combine chunks with proper resource management"""
        combined_df = None
        try:
            dfs = []
            for cf in chunk_files:
                if not cf.startswith("ERROR"):
                    df = pd.read_parquet(cf)
                    dfs.append(df)
                    os.remove(cf)
                else:
                    self.logger.error(f"Skipping error chunk: {cf}")
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # Save files with resource monitoring
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = []
                    
                    # Save parquet
                    futures.append(executor.submit(
                        combined_df.to_parquet,
                        f"output/{output_prefix}.parquet",
                        index=False,
                        compression='snappy'
                    ))
                    
                    # Save CSV
                    futures.append(executor.submit(
                        combined_df.to_csv,
                        f"output/{output_prefix}.csv",
                        index=False
                    ))
                    
                    # Wait for all saves to complete
                    for future in futures:
                        future.result()
            
        finally:
            # Cleanup
            if combined_df is not None:
                del combined_df
            gc.collect()

    def generate_data(self, num_records=100_000_000, seed=42):
        """Main data generation method with proper resource management"""
        start_time = datetime.now()
        
        try:
            os.makedirs('temp_chunks', exist_ok=True)
            os.makedirs('output', exist_ok=True)
            
            num_chunks = (num_records + self.chunk_size - 1) // self.chunk_size
            chunk_args = [(i, self.chunk_size, seed) for i in range(num_chunks)]
            
            self.logger.info(f"Starting data generation at: {start_time}")
            self.logger.info(f"Generating {num_records:,} records in {num_chunks} chunks")
            
            chunk_files = []
            with ProcessPoolExecutor(max_workers=self.cpu_count) as executor:
                try:
                    futures = list(tqdm(
                        executor.map(self.generate_chunk, chunk_args),
                        total=num_chunks,
                        desc="Generating chunks"
                    ))
                    chunk_files.extend(futures)
                except Exception as e:
                    self.logger.error(f"Error in data generation: {e}")
                finally:
                    executor.shutdown(wait=True)
            
            # Combine chunks
            CHUNKS_PER_COMBINE = max(10, self.cpu_count)
            for i in range(0, len(chunk_files), CHUNKS_PER_COMBINE):
                batch = chunk_files[i:i + CHUNKS_PER_COMBINE]
                if batch:
                    self.combine_chunks(batch, f"combined_{i//CHUNKS_PER_COMBINE}")
            
            # Save metadata
            end_time = datetime.now()
            metadata = {
                'num_records': num_records,
                'generation_time': str(end_time - start_time),
                'completion_time': end_time.isoformat()
            }
            
            with open('output/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error in data generation: {e}")
        finally:
            self.cleanup()
            self.logger.info(f"Process completed. Total time: {datetime.now() - start_time}")

def main():
    try:
        generator = PIDataGenerator()
        generator.generate_data(num_records=100_000_000)
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Cleaning up...")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Ensure all processes are terminated
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

if __name__ == "__main__":
    main()
