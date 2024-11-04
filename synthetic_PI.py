import pandas as pd 
import numpy as np
from faker import Faker
from datetime import datetime
import random
import uuid
import multiprocessing
from multiprocessing import shared_memory
import threading
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

class SystemResources:
    def __init__(self):
        self.cpu_physical_cores = psutil.cpu_count(logical=False)
        self.cpu_logical_cores = psutil.cpu_count(logical=True)
        self.total_memory = psutil.virtual_memory().total
        self.available_memory = psutil.virtual_memory().available
        self.usable_memory = int(self.available_memory * 0.9)  # 90% of available memory
        self.disk_usage = psutil.disk_usage('/')
        
    def __str__(self):
        return (
            f"Physical CPU cores: {self.cpu_physical_cores}\n"
            f"Logical CPU cores: {self.cpu_logical_cores}\n"
            f"Total Memory: {self.total_memory / (1024**3):.2f} GB\n"
            f"Available Memory: {self.available_memory / (1024**3):.2f} GB\n"
            f"Usable Memory: {self.usable_memory / (1024**3):.2f} GB\n"
            f"Disk Space Available: {self.disk_usage.free / (1024**3):.2f} GB"
        )

class PIDataGenerator:
    def __init__(self):
        self.system = SystemResources()
        self.setup_logging()
        self.setup_cleanup_handlers()
        self.setup_performance_parameters()
        
        # Print system information
        self.logger.info("System Resources:")
        self.logger.info(str(self.system))

    def setup_performance_parameters(self):
        """Calculate optimal performance parameters based on system resources"""
        # CPU utilization strategy
        self.num_processes = self.system.cpu_logical_cores
        self.threads_per_process = max(1, self.system.cpu_logical_cores // self.num_processes)
        
        # Memory allocation strategy
        memory_per_process = self.system.usable_memory // self.num_processes
        records_per_gb = 1_000_000  # Approximate number of records that fit in 1GB
        self.chunk_size = int((memory_per_process / (1024**3)) * records_per_gb)
        
        # I/O optimization
        self.io_workers = max(2, self.system.cpu_physical_cores // 2)
        
        self.logger.info(f"Performance Configuration:")
        self.logger.info(f"Processes: {self.num_processes}")
        self.logger.info(f"Threads per process: {self.threads_per_process}")
        self.logger.info(f"Chunk size: {self.chunk_size:,} records")
        self.logger.info(f"I/O workers: {self.io_workers}")

    def setup_logging(self):
        """Configure logging with performance metrics"""
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
        """Setup handlers for graceful shutdown"""
        def cleanup_handler(signum, frame):
            self.logger.info("Cleanup signal received. Terminating processes...")
            self.cleanup()
            exit(0)

        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGABRT]:
            signal.signal(sig, cleanup_handler)
        atexit.register(self.cleanup)

    def monitor_resources(self):
        """Monitor system resource usage"""
        process = psutil.Process()
        with process.oneshot():
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            num_threads = process.num_threads()
            
        memory_percent = psutil.virtual_memory().percent
        
        self.logger.info(
            f"Resource Usage - Memory: {memory_percent:.1f}%, "
            f"CPU: {cpu_percent:.1f}%, "
            f"Threads: {num_threads}"
        )
        
        return memory_info.rss, cpu_percent

    @staticmethod
    def generate_chunk(args):
        """Generate data chunk with thread and memory optimization"""
        chunk_id, chunk_size, seed, num_threads = args
        
        # Set process priority and affinity
        process = psutil.Process()
        process.nice(10)
        
        # Initialize generators
        fake = Faker()
        Faker.seed(seed + chunk_id)
        random.seed(seed + chunk_id)
        np.random.seed(seed + chunk_id)
        
        try:
            # Pre-allocate numpy arrays for better memory efficiency
            size = chunk_size
            genders = np.random.choice(['M', 'F', 'NB'], size=size)
            states = np.random.choice([fake.state() for _ in range(50)], size=size)
            
            # Generate data using multiple threads
            def generate_batch(start_idx, end_idx, results):
                batch_data = {
                    'id': [str(uuid.uuid4()) for _ in range(start_idx, end_idx)],
                    'ssn': [f"{random.randint(100,999):03d}-{random.randint(10,99):02d}-{random.randint(1000,9999):04d}" 
                           for _ in range(start_idx, end_idx)],
                    'full_name': [fake.name() for _ in range(start_idx, end_idx)],
                    'email': [fake.email() for _ in range(start_idx, end_idx)],
                    'phone': [f"{random.randint(100,999):03d}-{random.randint(100,999):03d}-{random.randint(1000,9999):04d}" 
                             for _ in range(start_idx, end_idx)],
                    'dob': [fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d') 
                           for _ in range(start_idx, end_idx)]
                }
                results.append(batch_data)

            # Split data generation across threads
            batch_size = size // num_threads
            threads = []
            results = []
            
            for i in range(num_threads):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size if i < num_threads - 1 else size
                thread = threading.Thread(
                    target=generate_batch,
                    args=(start_idx, end_idx, results)
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Combine results
            combined_data = {
                'id': [], 'ssn': [], 'full_name': [], 'email': [], 
                'phone': [], 'dob': [], 'gender': genders,
                'state': states
            }
            
            for batch_data in results:
                for key in batch_data:
                    combined_data[key].extend(batch_data[key])

            # Convert to DataFrame and save
            df = pd.DataFrame(combined_data)
            output_file = f"temp_chunks/chunk_{chunk_id}.parquet"
            df.to_parquet(output_file, index=False, compression='snappy')
            
            # Cleanup
            del df
            del combined_data
            del results
            gc.collect()
            
            return output_file
            
        except Exception as e:
            return f"ERROR: {str(e)}"

    def combine_chunks(self, chunk_files, output_prefix):
        """Combine chunks using parallel I/O"""
        try:
            # Read chunks in parallel
            with ThreadPoolExecutor(max_workers=self.io_workers) as executor:
                futures = [
                    executor.submit(pd.read_parquet, cf) 
                    for cf in chunk_files if not cf.startswith("ERROR")
                ]
                dfs = [future.result() for future in futures]

            # Combine DataFrames
            combined_df = pd.concat(dfs, ignore_index=True)

            # Save in multiple formats using parallel I/O
            with ThreadPoolExecutor(max_workers=self.io_workers) as executor:
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

            # Cleanup original chunk files
            for cf in chunk_files:
                if os.path.exists(cf):
                    os.remove(cf)

        except Exception as e:
            self.logger.error(f"Error combining chunks: {e}")
        finally:
            gc.collect()

    def generate_data(self, num_records=100_000_000, seed=42):
        """Main data generation method with resource optimization"""
        start_time = datetime.now()
        
        try:
            os.makedirs('temp_chunks', exist_ok=True)
            os.makedirs('output', exist_ok=True)
            
            num_chunks = (num_records + self.chunk_size - 1) // self.chunk_size
            chunk_args = [
                (i, self.chunk_size, seed, self.threads_per_process) 
                for i in range(num_chunks)
            ]
            
            self.logger.info(f"Starting data generation at: {start_time}")
            self.logger.info(f"Generating {num_records:,} records in {num_chunks} chunks")
            
            # Setup progress monitoring
            progress_bar = tqdm(total=num_chunks, desc="Generating chunks")
            
            # Generate chunks using process pool
            chunk_files = []
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                futures = {
                    executor.submit(self.generate_chunk, args): args 
                    for args in chunk_args
                }
                
                for future in concurrent.futures.as_completed(futures):
                    chunk_files.append(future.result())
                    progress_bar.update(1)
                    
                    # Monitor resources periodically
                    if len(chunk_files) % 10 == 0:
                        self.monitor_resources()
            
            progress_bar.close()
            
            # Combine chunks with optimized batch size
            self.logger.info("Combining chunks...")
            batch_size = max(20, self.num_processes * 2)
            for i in range(0, len(chunk_files), batch_size):
                batch = chunk_files[i:i + batch_size]
                if batch:
                    self.combine_chunks(batch, f"combined_{i//batch_size}")
                    self.monitor_resources()
            
            # Save metadata
            end_time = datetime.now()
            metadata = {
                'num_records': num_records,
                'num_chunks': num_chunks,
                'chunk_size': self.chunk_size,
                'processes_used': self.num_processes,
                'threads_per_process': self.threads_per_process,
                'generation_time': str(end_time - start_time),
                'system_resources': {
                    'cpu_cores': self.system.cpu_logical_cores,
                    'memory_gb': self.system.total_memory / (1024**3),
                    'peak_memory_used_gb': psutil.Process().memory_info().rss / (1024**3)
                }
            }
            
            with open('output/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error in data generation: {e}")
        finally:
            self.cleanup()
            self.logger.info(f"Process completed. Total time: {datetime.now() - start_time}")

    def cleanup(self):
        """Clean up resources and temporary files"""
        try:
            # Kill child processes
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

            # Remove temporary files
            if os.path.exists('temp_chunks'):
                for file in os.listdir('temp_chunks'):
                    try:
                        os.remove(os.path.join('temp_chunks', file))
                    except Exception as e:
                        self.logger.error(f"Error removing temp file: {e}")
                os.rmdir('temp_chunks')

            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

def main():
    try:
        generator = PIDataGenerator()
        generator.generate_data(num_records=100_000_000)
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Cleaning up...")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Final cleanup
        try:
            for proc in psutil.Process().children(recursive=True):
                proc.kill()
        except:
            pass

if __name__ == "__main__":
    main()
