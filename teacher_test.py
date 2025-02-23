import string
import sys
from students_code import words_in, lookup_word_count

# Read and process text file
def process_text_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Convert to lowercase and remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    words = text.translate(translator).lower().split()
    
    return words

# Main function to run the simulation
def main():
    # Read and process the input file
    filename = input("enter a file name")  # Change this to your text file name
    words_list = process_text_file(filename)

    # Run students' words_in function
    num_buckets, collisions = words_in(words_list)
    print(f"Number of Buckets Used: {num_buckets}")
    print(f"Collisions: {collisions}")

    # Convert list to a set for unique word lookups
    unique_words = set(words_list)

    # Run students' lookup_word_count function and track lookup score
    total_lookups = 0
    for word in unique_words:
        _, lookups = lookup_word_count(word)
        total_lookups += lookups

    # Calculate final score
    total_score = num_buckets + collisions + total_lookups

    # Print results
    print("\n--- Student's Hash Table Score ---")
    print(f"Buckets Used: {num_buckets}")
    print(f"Collisions: {collisions}")
    print(f"Total Lookups: {total_lookups}")
    print(f"Final Score: {total_score}")

if __name__ == "__main__":
    main()

