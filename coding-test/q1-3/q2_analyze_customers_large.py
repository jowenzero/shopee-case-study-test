import pandas as pd
from collections import Counter
from datetime import datetime
import psutil
import os

CHUNK_SIZE = 50000
csv_file = 'dataset/customers-2000000.csv'

# Get memory usage
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024

print("=" * 80)
print("LARGE CUSTOMER DATA ANALYSIS")
print("=" * 80)
print(f"\nProcessing in chunks of {CHUNK_SIZE:,} rows...")

# Initialize counters
total_rows = 0
peak_memory = mem_before
country_counter = Counter()
city_set = set()
year_counter = Counter()
yearmonth_counter = Counter()
company_counter = Counter()
email_set = set()
customer_id_set = set()
duplicate_emails = 0
duplicate_ids = 0
min_date = None
max_date = None

# Process CSV in chunks
for chunk_num, chunk in enumerate(pd.read_csv(csv_file, chunksize=CHUNK_SIZE), 1):
    print(f"Processing chunk {chunk_num}... ({chunk_num * CHUNK_SIZE:,} rows)", end='\r')

    total_rows += len(chunk)

    # Geographic data
    country_counter.update(chunk['Country'])
    city_set.update(chunk['City'].unique())

    # Temporal data
    chunk['Subscription Date'] = pd.to_datetime(chunk['Subscription Date'])
    year_counter.update(chunk['Subscription Date'].dt.year)
    yearmonth_counter.update(chunk['Subscription Date'].dt.to_period('M').astype(str))

    # Date range
    chunk_min = chunk['Subscription Date'].min()
    chunk_max = chunk['Subscription Date'].max()
    min_date = chunk_min if min_date is None else min(min_date, chunk_min)
    max_date = chunk_max if max_date is None else max(max_date, chunk_max)

    # Company data
    company_counter.update(chunk['Company'])

    # Duplicates check
    for email in chunk['Email']:
        if email in email_set:
            duplicate_emails += 1
        else:
            email_set.add(email)

    for cust_id in chunk['Customer Id']:
        if cust_id in customer_id_set:
            duplicate_ids += 1
        else:
            customer_id_set.add(cust_id)

    # Track peak memory
    current_memory = process.memory_info().rss / 1024 / 1024
    peak_memory = max(peak_memory, current_memory)

print("\n" + "=" * 80)

print("\n1. BASIC STATISTICS")
print("-" * 80)
print(f"Total customers: {total_rows:,}")
print(f"Date range: {min_date.date()} to {max_date.date()}")

print("\n2. GEOGRAPHIC DISTRIBUTION")
print("-" * 80)
print(f"Unique countries: {len(country_counter)}")
print(f"Unique cities: {len(city_set)}")
print(f"\nTop 10 countries by customer count:")
for country, count in country_counter.most_common(10):
    print(f"  {country}: {count:,}")

print("\n3. SUBSCRIPTION TIMELINE")
print("-" * 80)
print("Subscriptions by year:")
for year in sorted(year_counter.keys()):
    print(f"  {year}: {year_counter[year]:,}")

print("\nTop 10 months with most subscriptions:")
for month, count in sorted(yearmonth_counter.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {month}: {count:,}")

print("\n4. DATA DUPLICATES")
print("-" * 80)
print(f"Duplicate emails: {duplicate_emails}")
print(f"Duplicate customer IDs: {duplicate_ids}")

print("\n5. TOP COMPANIES")
print("-" * 80)
print("Top 10 companies by customer count:")
for company, count in company_counter.most_common(10):
    print(f"  {company}: {count}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

mem_after = process.memory_info().rss / 1024 / 1024
print("\nMEMORY USAGE")
print("-" * 80)
print(f"Peak memory during processing: {peak_memory:.2f} MB")
print(f"Memory used for loading CSV: {peak_memory - mem_before:.2f} MB")
print(f"Total process memory: {mem_after:.2f} MB")
