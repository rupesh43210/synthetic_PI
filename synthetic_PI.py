import pandas as pd 
import numpy as np
from faker import Faker
from datetime import datetime
import random
import uuid
import multiprocessing
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

class SystemResources:
    def __init__(self):
        """Dynamically detect and analyze system resources"""
        # CPU Detection
        self.cpu_physical_cores = psutil.cpu_count(logical=False)
        self.cpu_logical_cores = psutil.cpu_count(logical=True)
        self.cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory Detection
        mem = psutil.virtual_memory()
        self.total_memory = mem.total
        self.available_memory = mem.available
        self.memory_percent = mem.percent
        
        # Calculate usable resources
        self.calculate_optimal_resources()
        
        # I/O capabilities
        self.disk = psutil.disk_io_counters() if hasattr(psutil, 'disk_io_counters') else None
        self.disk_usage = psutil.disk_usage('/')
        
    def calculate_optimal_resources(self):
        """Calculate optimal resource allocation based on system capabilities"""
        # Calculate optimal process and thread distribution
        self.smt_ratio = self.cpu_logical_cores / self.cpu_physical_cores
        
        # Determine optimal process count (70-80% of logical cores)
        self.optimal_processes = max(1, int(self.cpu_logical_cores * 0.75))
        
        # Determine optimal threads per process
        self.optimal_threads = max(1, int(self.smt_ratio))
        
        # Calculate safe memory usage (90% of available memory)
        self.usable_memory = int(self.available_memory * 0.9)
        
        # Calculate I/O workers (25% of physical cores)
        self.optimal_io_workers = max(2, int(self.cpu_physical_cores * 0.25))
        
        # Calculate memory per worker
        total_workers = self.optimal_processes * self.optimal_threads
        self.memory_per_worker = self.usable_memory / total_workers
        
    def get_resource_summary(self):
        """Get formatted summary of system resources"""
        return {
            'cpu': {
                'physical_cores': self.cpu_physical_cores,
                'logical_cores': self.cpu_logical_cores,
                'smt_ratio': self.smt_ratio,
                'cpu_percent': self.cpu_percent
            },
            'memory': {
                'total_gb': self.total_memory / (1024**3),
                'available_gb': self.available_memory / (1024**3),
                'usable_gb': self.usable_memory / (1024**3),
                'memory_percent': self.memory_percent
            },
            'optimization': {
                'optimal_processes': self.optimal_processes,
                'optimal_threads': self.optimal_threads,
                'io_workers': self.optimal_io_workers,
                'memory_per_worker_gb': self.memory_per_worker / (1024**3)
            }
        }
    
    def __str__(self):
        summary = self.get_resource_summary()
        return (
            f"System Resources:\n"
            f"CPU:\n"
            f"  Physical Cores: {summary['cpu']['physical_cores']}\n"
            f"  Logical Cores: {summary['cpu']['logical_cores']}\n"
            f"  SMT Ratio: {summary['cpu']['smt_ratio']:.1f}\n"
            f"Memory:\n"
            f"  Total: {summary['memory']['total_gb']:.1f} GB\n"
            f"  Available: {summary['memory']['available_gb']:.1f} GB\n"
            f"  Usable: {summary['memory']['usable_gb']:.1f} GB\n"
            f"Optimization:\n"
            f"  Processes: {summary['optimization']['optimal_processes']}\n"
            f"  Threads per Process: {summary['optimization']['optimal_threads']}\n"
            f"  I/O Workers: {summary['optimization']['io_workers']}\n"
            f"  Memory per Worker: {summary['optimization']['memory_per_worker_gb']:.1f} GB"
        )

class PIDataGenerator:
    def __init__(self):
        """Initialize with dynamic system resource detection"""
        self.system = SystemResources()
        self.setup_logging()
        self.setup_cleanup_handlers()
        self.setup_performance_parameters()
        
        self.logger.info("Detected System Configuration:")
        self.logger.info(str(self.system))

    def setup_performance_parameters(self):
        """Setup performance parameters based on detected resources"""
        # Get optimal values from system analysis
        self.num_processes = self.system.optimal_processes
        self.threads_per_process = self.system.optimal_threads
        self.io_workers = self.system.optimal_io_workers
        
        # Calculate optimal chunk size based on available memory
        records_per_gb = 750_000  # Conservative estimate
        self.chunk_size = int((self.system.memory_per_worker / (1024**3)) * records_per_gb)
        
        # Ensure chunk size is within reasonable bounds
        self.chunk_size = max(100_000, min(self.chunk_size, 2_000_000))
        
        # Log configuration
        self.logger.info("Performance Configuration:")
        self.logger.info(f"Processes: {self.num_processes}")
        self.logger.info(f"Threads per process: {self.threads_per_process}")
        self.logger.info(f"Total workers: {self.num_processes * self.threads_per_process}")
        self.logger.info(f"Chunk size: {self.chunk_size:,} records")
        self.logger.info(f"I/O workers: {self.io_workers}")

    def setup_logging(self):
        """Configure logging with timestamp and performance metrics"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pi_generator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def monitor_resources(self):
        """Monitor current resource usage"""
        process = psutil.Process()
        with process.oneshot():
            # CPU Usage
            cpu_percent = process.cpu_percent()
            cpu_times = process.cpu_times()
            
            # Memory Usage
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Thread Count
            thread_count = process.num_threads()
            
            # I/O Counters
            io_counters = process.io_counters()
            
            self.logger.info(
                f"Resource Usage:\n"
                f"  CPU: {cpu_percent:.1f}%\n"
                f"  Memory: {memory_info.rss / (1024**3):.1f} GB ({memory_percent:.1f}%)\n"
                f"  Threads: {thread_count}\n"
                f"  I/O Read: {io_counters.read_bytes / (1024**2):.1f} MB\n"
                f"  I/O Write: {io_counters.write_bytes / (1024**2):.1f} MB"
            )
            
        return memory_info.rss, cpu_percent

    @staticmethod
    def generate_chunk(args):
        """Generate data chunk with dynamic thread optimization"""
        chunk_id, chunk_size, seed, num_threads = args
        
        # Set process priority
        process = psutil.Process()
        process.nice(10)
        
        # Initialize generators
        fake = Faker()
        Faker.seed(seed + chunk_id)
        random.seed(seed + chunk_id)
        np.random.seed(seed + chunk_id)
        
        try:
            # Calculate batch size based on thread count
            batch_size = chunk_size // num_threads
            
            # Shared data arrays
            genders = np.random.choice(['M', 'F', 'NB'], size=chunk_size)
            states = np.random.choice([fake.state() for _ in range(50)], size=chunk_size)
            
            # Thread-safe results container
            results = []
            
            def generate_batch(start_idx, end_idx):
                """Generate a batch of data within a thread"""
                return {
                    'id': [str(uuid.uuid4()) for _ in range(start_idx, end_idx)],
                    'ssn': [f"{random.randint(100,999):03d}-{random.randint(10,99):02d}-{random.randint(1000,9999):04d}" 
                           for _ in range(start_idx, end_idx)],
                    'full_name': [fake.name() for _ in range(start_idx, end_idx)],
                    'email': [fake.email() for _ in range(start_idx, end_idx)],
                    'phone': [f"{random.randint(100,999):03d}-{random.randint(100,999):03d}-{random.randint(1000,9999):04d}" 
                             for _ in range(start_idx, end_idx)],
                    'dob': [fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d') 
                           for _ in range(start_idx, end_idx)],
                    'gender': genders[start_idx:end_idx],
                    'state': states[start_idx:end_idx]
                }
            
            # Create and run threads
            threads = []
            for i in range(num_threads):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size if i < num_threads - 1 else chunk_size
                
                thread = threading.Thread(
                    target=lambda: results.append(generate_batch(start_idx, end_idx))
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            # Combine results
            combined_data = {}
            for key in results[0].keys():
                combined_data[key] = []
                for batch in results:
                    combined_data[key].extend(batch[key])
            
            # Create DataFrame and save
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

    def generate_data(self, num_records=100_000_000, seed=42):
        """Main data generation method with dynamic optimization"""
        start_time = datetime.now()
        
        try:
            os.makedirs('temp_chunks', exist_ok=True)
            os.makedirs('output', exist_ok=True)
            
            # Calculate chunks based on optimal size
            num_chunks = (num_records + self.chunk_size - 1) // self.chunk_size
            chunk_args = [
                (i, self.chunk_size, seed, self.threads_per_process) 
                for i in range(num_chunks)
            ]
            
            self.logger.info(f"Starting data generation at: {start_time}")
            self.logger.info(f"Generating {num_records:,} records in {num_chunks} chunks")
            
            # Generate chunks with progress monitoring
            chunk_files = []
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                futures = {
                    executor.submit(self.generate_chunk, args): args[0] 
                    for args in chunk_args
                }
                
                with tqdm(total=len(futures), desc="Generating chunks") as progress:
                    for future in as_completed(futures):
                        result = future.result()
                        chunk_files.append(result)
                        progress.update(1)
                        
                        # Monitor resources periodically
                        if len(chunk_files) % 5 == 0:
                            self.monitor_resources()
            
            # Combine chunks with dynamic batch size
            self.logger.info("Combining chunks...")
            batch_size = self.io_workers * 2
            
            for i in range(0, len(chunk_files), batch_size):
                batch = chunk_files[i:i + batch_size]
                if batch:
                    self.combine_chunks(batch, f"combined_{i//batch_size}")
                    gc.collect()
            
            # Generate final metadata
            end_time = datetime.now()
            duration = end_time - start_time
            
            metadata = {
                'execution': {
                    'num_records': num_records,
                    'num_chunks': num_chunks,
                    'chunk_size': self.chunk_size,
                    'generation_time_seconds': duration.total_seconds(),
                    'records_per_second': num_records / duration.total_seconds()
                },
                'system_configuration': self.system.get_resource_summary(),
                'performance_metrics': {
                    'processes_used': self.num_processes,
                    'threads_per_process': self.threads_per_process,
                    'io_workers': self.io_workers,
                    'peak_memory_gb': psutil.Process().memory_info().rss / (1024**3)
                }
            }
            
            with open('output/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error in data generation: {str(e)}")
            raise
        finally:
            self.cleanup()
            self.logger.info(f"Process completed. Total time: {duration}")

    def cleanup(self):
        """Clean up resources and temporary files"""
        try:
            # Terminate child processes
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
        print(f"Error in main: {str(e)}")
    finally:
        # Final cleanup
        try:
            for proc in psutil.Process().children(recursive=True):
                proc.kill()
        except:
            pass

if __name__ == "__main__":
    main()
