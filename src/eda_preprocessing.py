import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = DATA_DIR 
COMPLAINTS_FILE = os.path.join(DATA_DIR, 'complaints.csv')
FILTERED_COMPLAINTS_FILE = os.path.join(OUTPUT_DIR, 'filtered_complaints.csv')


os.makedirs(DATA_DIR, exist_ok=True)


TARGET_PRODUCT_KEYWORDS = [
    'credit card',
    'personal loan',
    'buy now, pay later', # Exact match for BNPL
    'savings account',
    'money transfer',
    # Common variations or related products that should also be included
    'checking or savings account',
    'payday loan',
    'title loan', # Often grouped with personal loans
    'consumer loan', # A general category that might include personal loans
    'money service', # Broader term often used for money transfers (e.g., 'Money transfer, virtual currency, or money service')
    'virtual currency' # Related to money transfers
]

# --- 1.1: Load the full CFPB complaint dataset ---
print(f"Loading data from {COMPLAINTS_FILE}...")
try:
    # Use low_memory=False to prevent DtypeWarning for columns with mixed types
    df = pd.read_csv(COMPLAINTS_FILE, low_memory=False)
    print(f"Successfully loaded {len(df)} records.")
except FileNotFoundError:
    print(f"ERROR: {COMPLAINTS_FILE} not found.")
    print(f"Please ensure the CSV file is in the '{DATA_DIR}' directory.")
    exit() # Exit the script if the data file is not found

# --- 1.2: Perform an initial EDA to understand the data ---
print("\n--- Initial Data Exploration (EDA) ---")
print("\nDataFrame Info:")
df.info() # Provides a concise summary of the DataFrame, including data types and non-null values

print("\nFirst 5 rows of the dataset:")
print(df.head()) # Shows the first few rows to get a sense of the data structure

print("\nDescriptive Statistics for numerical columns:")
print(df.describe()) # Provides statistical summary for numerical columns

print("\nNumber of unique values per column:")
print(df.nunique()) # Shows how many unique values are in each column

print("\nMissing values before filtering and cleaning:")
# Displays columns with any missing values and their counts, sorted in descending order
print(df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False))

# --- 1.3: Analyze the distribution of complaints across different Products ---
print("\n--- Product Distribution Analysis ---")
print("\nDistribution of complaints across 'Product' (Top 20 products before filtering):")
product_distribution = df['Product'].value_counts()
print(product_distribution.head(20)) # Show top 20 products for a quick overview

plt.figure(figsize=(12, 8))
sns.barplot(y=product_distribution.head(20).index, x=product_distribution.head(20).values, palette='viridis')
plt.title('Top 20 Product Categories by Complaint Count (Before Filtering)')
plt.xlabel('Number of Complaints')
plt.ylabel('Product')
plt.tight_layout()
# Save the plot for your report
plt.savefig(os.path.join(OUTPUT_DIR, 'product_distribution_initial.png'))
plt.show() # Display the plot

# --- 1.4: Calculate and visualize the length (word count) of the Consumer complaint narrative ---
print("\n--- Consumer Complaint Narrative Length Analysis ---")
# Ensure 'Consumer complaint narrative' column exists and convert its values to string type.
# Replace 'nan' string representation with an empty string for easier processing.
if 'Consumer complaint narrative' in df.columns:
    df['Consumer complaint narrative'] = df['Consumer complaint narrative'].astype(str).replace('nan', '')
else:
    print("WARNING: 'Consumer complaint narrative' column not found. Creating an empty column.")
    df['Consumer complaint narrative'] = '' # Create an empty column to prevent errors later

# Calculate word count for each narrative.
# If the narrative is just whitespace, .strip() makes it empty, resulting in 0 words.
df['narrative_word_count'] = df['Consumer complaint narrative'].apply(lambda x: len(x.split()) if x.strip() else 0)

print(f"\nAverage narrative word count: {df['narrative_word_count'].mean():.2f}")
print(f"Median narrative word count: {df['narrative_word_count'].median():.2f}")
print(f"Min narrative word count: {df['narrative_word_count'].min()}")
print(f"Max narrative word count: {df['narrative_word_count'].max()}")

plt.figure(figsize=(10, 6))
# Plot a histogram of word counts, excluding narratives with 0 words for clearer visualization
sns.histplot(df['narrative_word_count'][df['narrative_word_count'] > 0], bins=50, kde=True)
plt.title('Distribution of Consumer Complaint Narrative Length (Word Count)')
plt.xlabel('Word Count')
plt.ylabel('Number of Complaints')
plt.tight_layout()
# Save the plot
plt.savefig(os.path.join(OUTPUT_DIR, 'narrative_length_distribution.png'))
plt.show()

# Identify and count complaints with very short (but non-zero) or very long narratives
short_narratives_count = len(df[(df['narrative_word_count'] > 0) & (df['narrative_word_count'] < 5)])
long_narratives_count = len(df[df['narrative_word_count'] > 500])
print(f"\nNumber of complaints with non-zero but less than 5 words: {short_narratives_count}")
print(f"Number of complaints with more than 500 words: {long_narratives_count}")


# --- 1.5: Identify the number of complaints with and without narratives ---
print("\n--- Missing/Empty Narratives Analysis ---")
initial_total_complaints = len(df)
# Count narratives that effectively have no content (0 words after stripping whitespace)
complaints_with_zero_words = len(df[df['narrative_word_count'] == 0])
complaints_with_non_empty_narrative = initial_total_complaints - complaints_with_zero_words

print(f"Total complaints initially loaded: {initial_total_complaints}")
print(f"Complaints with empty/missing narrative (word count 0): {complaints_with_zero_words}")
print(f"Complaints with non-empty narrative: {complaints_with_non_empty_narrative}")
print(f"Percentage of complaints with non-empty narrative: {complaints_with_non_empty_narrative / initial_total_complaints * 100:.2f}%")


# --- 1.6: Filter the dataset ---
print("\n--- Filtering Data ---")

# Step 1: Remove records where the 'Consumer complaint narrative' is empty (word count 0)
df_filtered = df[df['narrative_word_count'] > 0].copy()
print(f"Complaints after removing empty narratives: {len(df_filtered)}")

# Step 2: Filter by CrediTrust's target products.
# Create a lowercase version of the 'Product' column for case-insensitive matching
df_filtered['Product_lower'] = df_filtered['Product'].str.lower()

# Create a boolean mask: a row is True if its 'Product_lower' column contains any of the TARGET_PRODUCT_KEYWORDS
product_mask = df_filtered['Product_lower'].apply(
    lambda x: any(keyword in x for keyword in TARGET_PRODUCT_KEYWORDS)
)
df_filtered = df_filtered[product_mask].copy()

print(f"Complaints after filtering for target products: {len(df_filtered)}")
print("\nDistribution of complaints across FILTERED 'Product' categories:")
filtered_product_distribution = df_filtered['Product'].value_counts()
print(filtered_product_distribution)

plt.figure(figsize=(12, 8))
sns.barplot(y=filtered_product_distribution.index, x=filtered_product_distribution.values, palette='viridis')
plt.title('Product Categories by Complaint Count (After Filtering)')
plt.xlabel('Number of Complaints')
plt.ylabel('Product')
plt.tight_layout()
# Save the filtered product distribution plot
plt.savefig(os.path.join(OUTPUT_DIR, 'product_distribution_filtered.png'))
plt.show()


# --- 1.7: Clean the text narratives ---
print("\n--- Cleaning Text Narratives ---")

def clean_text(text):
    """
    Performs a series of text cleaning operations on a given string.
    - Lowercases text.
    - Removes content within brackets, parentheses, and curly braces (e.g., [redacted], (info)).
    - Removes common boilerplate phrases that don't add semantic value.
    - Removes sensitive identifiers (like 'XXXX') and isolated 'x' characters.
    - Removes numbers.
    - Removes punctuation.
    - Removes extra whitespace.
    """
    text = str(text).lower()  # Lowercasing

    # Remove content within brackets, parentheses, or curly braces
    text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', text)

    # Remove common boilerplate phrases. These are often generic and don't contribute to complaint specifics.
    boilerplate_patterns = [
        r"i am writing to (file a complaint|complain about|express my concern)",
        r"to whom it may concern",
        r"this is a complaint regarding",
        r"i want to report",
        r"i would like to report",
        r"i would like to file a complaint",
        r"i am writing this letter to",
        r"i am writing regarding",
        r"i wish to express my dissatisfaction",
        r"i am contacting you regarding",
        r"i am writing in reference to",
        r"this complaint is about",
        r"please investigate",
        r"consumer complaint narrative", # Sometimes this exact phrase appears in the text
        r"sent to consumer finance protection bureau",
        r"re : account", # Common redaction prefix
        r"reguarding account",
        r"my account", r"my loan", r"my card", r"my savings", # Generic "my X" phrases
        r"cfpb id" # Specific CFPB ID mentions
    ]
    # Apply boilerplate removal in a case-insensitive manner
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Remove sensitive identifiers/redactions like "xxxx", "xxxxxxxx"
    text = re.sub(r'x{2,}', '', text) # Removes two or more consecutive 'x' characters

    # Remove isolated single 'x' characters if they appear to be redactions or meaningless
    text = re.sub(r'\b x \b', ' ', text)

    # Remove numbers (as they are often account numbers, dates, or other non-semantic info for complaint themes)
    text = re.sub(r'\d+', '', text)

    # Remove punctuation. Keep only letters and spaces.
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra whitespaces (multiple spaces to single space, and strip leading/trailing)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply the cleaning function to the 'Consumer complaint narrative' column
df_filtered['cleaned_narrative'] = df_filtered['Consumer complaint narrative'].apply(clean_text)

# Final check for empty narratives after extensive cleaning
# Some narratives might become empty if they were only noise, boilerplate, or redactions
df_final = df_filtered[df_filtered['cleaned_narrative'].str.strip() != ''].copy()
print(f"\nComplaints after final check for empty cleaned narratives: {len(df_final)}")


# Re-evaluate word count for the cleaned narratives to see the effect of cleaning
df_final['cleaned_narrative_word_count'] = df_final['cleaned_narrative'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)

print(f"\nAverage cleaned narrative word count: {df_final['cleaned_narrative_word_count'].mean():.2f}")
print(f"Median cleaned narrative word count: {df_final['cleaned_narrative_word_count'].median():.2f}")
print(f"Min cleaned narrative word count: {df_final['cleaned_narrative_word_count'].min()}")
print(f"Max cleaned narrative word count: {df_final['cleaned_narrative_word_count'].max()}")


# Display some examples of original vs. cleaned narratives for manual inspection
print("\n--- Original vs Cleaned Narratives (Examples) ---")
# Get 5 random samples to see a variety of cleaning effects
sample_df = df_final.sample(min(5, len(df_final)), random_state=42) # Use random_state for reproducibility
for i, row in sample_df.iterrows():
    original = row['Consumer complaint narrative']
    cleaned = row['cleaned_narrative']
    print(f"\n--- Example from Original Complaint ID: {row['Complaint ID']} ---")
    print(f"Original: {original[:500]}...") # Print first 500 characters
    print(f"Cleaned:  {cleaned[:500]}...") # Print first 500 characters


# Select only the columns that are relevant for subsequent steps or useful for context.
# 'Complaint ID' is crucial for tracing back to the original complaint.
columns_to_keep = [
    'Date received', 'Product', 'Sub-product', 'Issue', 'Sub-issue',
    'Company', 'State', 'ZIP code', 'Date sent to company',
    'Company response to consumer', 'Timely response?', 'Consumer disputed?',
    'Consumer complaint narrative', # Keep original narrative for reference/debugging
    'cleaned_narrative', # This is the primary column for embedding and RAG
    'Complaint ID' # Critical for tracing chunks back to original complaint
]

# Filter the DataFrame to include only the desired columns.
# This loop handles cases where some columns might be missing in a specific dataset version.
df_final = df_final[[col for col in columns_to_keep if col in df_final.columns]].copy()

# Ensure 'Complaint ID' is set as the DataFrame index.
# This makes it easy to associate chunks with their original complaint ID later.
if 'Complaint ID' in df_final.columns:
    df_final.set_index('Complaint ID', inplace=True)
else:
    # Fallback: if 'Complaint ID' column is missing, generate a new one and set it as index.
    df_final['Complaint ID'] = range(1, len(df_final) + 1)
    df_final.set_index('Complaint ID', inplace=True)
    print("WARNING: 'Complaint ID' column not found. A new 'Complaint ID' has been generated and set as index.")


# --- Deliverable: Save the cleaned and filtered dataset ---
print(f"\n--- Saving filtered and cleaned data to {FILTERED_COMPLAINTS_FILE} ---")
# Save the DataFrame to a new CSV file, including the index (Complaint ID)
df_final.to_csv(FILTERED_COMPLAINTS_FILE, index=True)
print("Data saved successfully.")

print("\n--- Task 1: EDA and Data Preprocessing Completed ---")