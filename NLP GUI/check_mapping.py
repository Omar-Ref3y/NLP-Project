import pickle
import os

# Path to the mapping file
mapping_path = os.path.join("data", "combined_mapping.pkl")

# Load the mapping file
print(f"Loading mapping file from: {mapping_path}")
try:
    with open(mapping_path, 'rb') as f:
        mapping_data = pickle.load(f)
    
    # Check if it's a dictionary
    if isinstance(mapping_data, dict):
        print(f"Mapping data is a dictionary with {len(mapping_data)} items")
        print(f"Keys: {list(mapping_data.keys()) if len(mapping_data.keys()) < 20 else 'Too many keys to display'}")
        
        # If it has more than 100 keys, it's likely a direct mapping
        if len(mapping_data) > 100:
            print("This appears to be a direct mapping (idx -> word or word -> idx)")
            # Show a few sample items
            items = list(mapping_data.items())[:5]
            print(f"Sample items: {items}")
    else:
        print(f"Mapping data is not a dictionary, it's a {type(mapping_data)}")
        print(f"Content: {mapping_data}")
except Exception as e:
    print(f"Error loading mapping file: {str(e)}")
    
# Also check the tokenizer
tokenizer_path = os.path.join("data", "tokenizer.pkl")
print(f"\nLoading tokenizer from: {tokenizer_path}")
try:
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded successfully")
    print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")
    print(f"Sample word indices: {list(tokenizer.word_index.items())[:5]}")
except Exception as e:
    print(f"Error loading tokenizer: {str(e)}")
