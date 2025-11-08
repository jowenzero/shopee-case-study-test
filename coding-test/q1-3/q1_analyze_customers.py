import pandas as pd
from datetime import datetime
import psutil
import os

# Get memory usage
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024

# Load data
df = pd.read_csv('dataset/customers-100000.csv')

mem_after = process.memory_info().rss / 1024 / 1024
mem_used = mem_after - mem_before

print("=" * 80)
print("CUSTOMER DATA ANALYSIS")
print("=" * 80)

print("\n1. BASIC STATISTICS")
print("-" * 80)
print(f"Total customers: {len(df):,}")
print(f"Date range: {df['Subscription Date'].min()} to {df['Subscription Date'].max()}")
print(f"\nColumns: {', '.join(df.columns)}")
print(f"\nMissing values per column:")
print(df.isnull().sum())

print("\n2. GEOGRAPHIC DISTRIBUTION")
print("-" * 80)
print(f"Unique countries: {df['Country'].nunique()}")
print(f"Unique cities: {df['City'].nunique()}")
print(f"\nTop 10 countries by customer count:")
country_counts = df['Country'].value_counts().head(10)
for country, count in country_counts.items():
    print(f"  {country}: {count:,}")

print("\n3. SUBSCRIPTION TIMELINE")
print("-" * 80)
df['Subscription Date'] = pd.to_datetime(df['Subscription Date'])
df['Year'] = df['Subscription Date'].dt.year
df['Month'] = df['Subscription Date'].dt.month
df['YearMonth'] = df['Subscription Date'].dt.to_period('M')

print("Subscriptions by year:")
year_counts = df['Year'].value_counts().sort_index()
for year, count in year_counts.items():
    print(f"  {year}: {count:,}")

print("\nTop 10 months with most subscriptions:")
month_counts = df['YearMonth'].value_counts().head(10)
for month, count in month_counts.items():
    print(f"  {month}: {count:,}")

print("\n4. DATA DUPLICATES")
print("-" * 80)
duplicate_emails = df['Email'].duplicated().sum()
duplicate_ids = df['Customer Id'].duplicated().sum()
print(f"Duplicate emails: {duplicate_emails}")
print(f"Duplicate customer IDs: {duplicate_ids}")

print("\n5. TOP COMPANIES")
print("-" * 80)
print("Top 10 companies by customer count:")
company_counts = df['Company'].value_counts().head(10)
for company, count in company_counts.items():
    print(f"  {company}: {count}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print("\nMEMORY USAGE")
print("-" * 80)
print(f"Memory used for loading CSV: {mem_used:.2f} MB")
print(f"Total process memory: {mem_after:.2f} MB")
