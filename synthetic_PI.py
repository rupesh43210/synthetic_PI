import pandas as pd
from faker import Faker
import random

# Initialize Faker instance
fake = Faker()

# Set total number of rows and batch size
total_rows = 1000000  # Target total records
batch_size = 100000  # Adjust for memory efficiency

# Define an extensive set of complex PI columns
columns = [
    "Full Name", "Gender", "Email", "Phone Number", "Address", "City", "State",
    "Country", "Postal Code", "Date of Birth", "SSN", "Credit Card Number",
    "Credit Card Expiry", "Credit Card Provider", "Bank Account Number",
    "Driver's License", "Vehicle Registration", "Company", "Job Title", "Username",
    "Password", "IP Address", "MAC Address", "Device ID", "Browser User Agent",
    "Passport Number", "Nationality", "Education Level", "Marital Status",
    "Income", "Favorite Color", "Blood Type", "Height", "Weight", "Insurance Policy",
    "Policy Expiry", "Insurance Provider", "Tax ID", "Bank Name", "Occupation",
    "Transaction ID", "Transaction Date", "Transaction Amount", "Transaction Type",
    "Medical Record Number", "Allergies", "Blood Pressure", "Heart Rate", 
    "Cholesterol Level", "Genetic Test Results", "Emergency Contact", 
    "Emergency Contact Relation", "Family Medical History", "Biometric ID"
]

# Generate a list of transaction types for added variability
transaction_types = ["Purchase", "Refund", "Withdrawal", "Deposit", "Transfer"]

# Generate random genetic test results
genetic_tests = ["Positive", "Negative", "Inconclusive"]

# Family medical history types
family_medical_history = ["Diabetes", "Heart Disease", "Cancer", "None"]

# Function to generate a batch of synthetic data with diverse PI attributes
def generate_data_batch(batch_size):
    data = {
        "Full Name": [fake.name() for _ in range(batch_size)],
        "Gender": [fake.random_element(elements=("Male", "Female", "Non-Binary", "Other")) for _ in range(batch_size)],
        "Email": [fake.email() for _ in range(batch_size)],
        "Phone Number": [fake.phone_number() for _ in range(batch_size)],
        "Address": [fake.address() for _ in range(batch_size)],
        "City": [fake.city() for _ in range(batch_size)],
        "State": [fake.state() for _ in range(batch_size)],
        "Country": [fake.country() for _ in range(batch_size)],
        "Postal Code": [fake.postcode() for _ in range(batch_size)],
        "Date of Birth": [fake.date_of_birth(minimum_age=18, maximum_age=90) for _ in range(batch_size)],
        "SSN": [fake.ssn() for _ in range(batch_size)],
        "Credit Card Number": [fake.credit_card_number() for _ in range(batch_size)],
        "Credit Card Expiry": [fake.credit_card_expire() for _ in range(batch_size)],
        "Credit Card Provider": [fake.credit_card_provider() for _ in range(batch_size)],
        "Bank Account Number": [fake.bban() for _ in range(batch_size)],
        "Driver's License": [fake.license_plate() for _ in range(batch_size)],
        "Vehicle Registration": [fake.bothify(text="??-####") for _ in range(batch_size)],
        "Company": [fake.company() for _ in range(batch_size)],
        "Job Title": [fake.job() for _ in range(batch_size)],
        "Username": [fake.user_name() for _ in range(batch_size)],
        "Password": [fake.password() for _ in range(batch_size)],
        "IP Address": [fake.ipv4() for _ in range(batch_size)],
        "MAC Address": [fake.mac_address() for _ in range(batch_size)],
        "Device ID": [fake.uuid4() for _ in range(batch_size)],
        "Browser User Agent": [fake.user_agent() for _ in range(batch_size)],
        "Passport Number": [fake.bothify(text="P#######") for _ in range(batch_size)],
        "Nationality": [fake.country() for _ in range(batch_size)],
        "Education Level": [fake.random_element(elements=("High School", "Bachelor's", "Master's", "PhD")) for _ in range(batch_size)],
        "Marital Status": [fake.random_element(elements=("Single", "Married", "Divorced", "Widowed")) for _ in range(batch_size)],
        "Income": [fake.random_int(min=30000, max=200000) for _ in range(batch_size)],
        "Favorite Color": [fake.color_name() for _ in range(batch_size)],
        "Blood Type": [fake.random_element(elements=("A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-")) for _ in range(batch_size)],
        "Height": [fake.random_int(min=150, max=200) for _ in range(batch_size)],  # in cm
        "Weight": [fake.random_int(min=50, max=120) for _ in range(batch_size)],  # in kg
        "Insurance Policy": [fake.bothify(text="INS-#####") for _ in range(batch_size)],
        "Policy Expiry": [fake.date_this_century(before_today=False, after_today=True) for _ in range(batch_size)],
        "Insurance Provider": [fake.company() for _ in range(batch_size)],
        "Tax ID": [fake.ssn() for _ in range(batch_size)],
        "Bank Name": [fake.company() for _ in range(batch_size)],
        "Occupation": [fake.job() for _ in range(batch_size)],
        "Transaction ID": [fake.uuid4() for _ in range(batch_size)],
        "Transaction Date": [fake.date_this_year() for _ in range(batch_size)],
        "Transaction Amount": [round(random.uniform(10, 10000), 2) for _ in range(batch_size)],
        "Transaction Type": [fake.random_element(elements=transaction_types) for _ in range(batch_size)],
        "Medical Record Number": [fake.bothify(text="MRN-######") for _ in range(batch_size)],
        "Allergies": [fake.random_element(elements=("None", "Peanuts", "Dust", "Pollen", "Gluten")) for _ in range(batch_size)],
        "Blood Pressure": [f"{random.randint(90, 140)}/{random.randint(60, 90)}" for _ in range(batch_size)],
        "Heart Rate": [random.randint(60, 100) for _ in range(batch_size)],
        "Cholesterol Level": [random.randint(150, 240) for _ in range(batch_size)],
        "Genetic Test Results": [fake.random_element(elements=genetic_tests) for _ in range(batch_size)],
        "Emergency Contact": [fake.name() for _ in range(batch_size)],
        "Emergency Contact Relation": [fake.random_element(elements=("Parent", "Sibling", "Spouse", "Friend", "Guardian")) for _ in range(batch_size)],
        "Family Medical History": [fake.random_element(elements=family_medical_history) for _ in range(batch_size)],
        "Biometric ID": [fake.bothify(text="BIO-#####") for _ in range(batch_size)]
    }
    return pd.DataFrame(data)

# Save each batch to separate Excel files to manage memory
for i in range(total_rows // batch_size):
    batch_df = generate_data_batch(batch_size)
    output_file = f"complex_synthetic_pi_data_batch_{i + 1}.xlsx"
    batch_df.to_excel(output_file, index=False)
    print(f"Batch {i + 1} saved to {output_file}")

print("Data generation complete.")
