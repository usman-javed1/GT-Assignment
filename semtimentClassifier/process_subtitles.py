import pandas as pd
import os
import re

def parse_timestamp(line):
    # Parse timestamp in format [123.45]
    match = re.match(r'\[(\d+\.\d+)\](.+)', line)
    if match:
        timestamp = float(match.group(1))
        text = match.group(2).strip()
        return timestamp, text
    return None, None

def process_subtitle_file(file_path):
    print(f"Processing file: {file_path}")
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"Successfully read {len(lines)} lines from {file_path}")
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return []
    
    subtitles = []
    for line in lines:
        line = line.strip()
        if line:
            timestamp, text = parse_timestamp(line)
            if timestamp is not None:
                subtitles.append({
                    'timestamp': timestamp,
                    'text': text
                })
    
    print(f"Extracted {len(subtitles)} subtitles from {file_path}")
    return subtitles

def format_timestamp(seconds):
    # Convert seconds to HH:MM:SS format
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def main():
    # Read the metadata file
    metadata_path = '/home/usman/Desktop/GT Assignment /semtimentClassifier/drama_metadata_filtered.xlsx'
    print(f"Reading metadata from: {metadata_path}")
    
    try:
        metadata_df = pd.read_excel(metadata_path)
        print(f"Successfully read metadata file with {len(metadata_df)} rows")
    except Exception as e:
        print(f"Error reading metadata file: {str(e)}")
        return
    
    base_path = '/home/usman/Desktop/GT Assignment /farzan__transcripts'
    output_dir = '/home/usman/Desktop/GT Assignment /semtimentClassifier/processed_subtitles'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # List to store all subtitle pairs for master file
    all_subtitle_pairs = []
    
    for idx, row in metadata_df.iterrows():
        print(f"\nProcessing drama {idx + 1}/{len(metadata_df)}")
        drama_name = row['Drama Name']
        urdu_file = row['Urdu Subtitles with Timestamp File Name']
        english_file = row['English Subtitles with Timestamp File Name']
        urdu_file = urdu_file.replace('Ep_', 'Ep')
        english_file = english_file.replace('Ep_', 'Ep')
        
        print(f"Drama: {drama_name}")
        print(f"Urdu file: {urdu_file}")
        print(f"English file: {english_file}")
        
        # Extract episode number and URL from metadata
        episode_no = row.get('Episode No#', row.get('Episode', ''))
        url = row.get('Youtube Link', row.get('URL', ''))
        
        # Process both subtitle files
        drama_base_path = os.path.join(base_path, drama_name)
        urdu_path = os.path.join(drama_base_path, urdu_file)
        english_path = os.path.join(drama_base_path, english_file)
        
        print(f"Full Urdu path: {urdu_path}")
        print(f"Full English path: {english_path}")
        
        urdu_subtitles = process_subtitle_file(urdu_path)
        english_subtitles = process_subtitle_file(english_path)
        
        if not urdu_subtitles or not english_subtitles:
            print(f"Skipping {drama_name} due to missing subtitles")
            continue
        
        # Create a dictionary to match subtitles by timestamp
        subtitle_pairs = []
        sentence_no = 1
        for urdu_sub in urdu_subtitles:
            # Find matching English subtitle with closest timestamp
            closest_eng = min(english_subtitles, 
                            key=lambda x: abs(x['timestamp'] - urdu_sub['timestamp']))
            
            # Only match if timestamps are within 1 second of each other
            if abs(closest_eng['timestamp'] - urdu_sub['timestamp']) <= 1.0:
                pair = {
                    'Drama Name': drama_name,
                    'Episode No#': episode_no,
                    'URL': url,
                    'Sentence No': sentence_no,
                    'Timestamps': format_timestamp(urdu_sub['timestamp']),
                    'Urdu Sentence': urdu_sub['text'],
                    'English Sentence': closest_eng['text']
                }
                subtitle_pairs.append(pair)
                all_subtitle_pairs.append(pair)  # Add to master list
                sentence_no += 1
        
        if subtitle_pairs:
            # Create DataFrame and save individual drama CSV
            output_df = pd.DataFrame(subtitle_pairs)
            output_file = os.path.join(output_dir, f'{drama_name}_aligned_subtitles.csv')
            try:
                if os.path.exists(output_file):
                    existing_df = pd.read_csv(output_file)
                    output_df = pd.concat([existing_df, output_df], ignore_index=True)
                    # Drop duplicates based on Drama Name, Episode No#, URL, Sentence No, Urdu Sentence, English Sentence
                    output_df = output_df.drop_duplicates(subset=['Drama Name', 'Episode No#', 'URL', 'Sentence No', 'Urdu Sentence', 'English Sentence'])
                output_df.to_csv(output_file, index=False, encoding='utf-8')
                print(f"Successfully updated {output_file} with {len(output_df)} subtitle pairs")
            except Exception as e:
                print(f"Error saving CSV file {output_file}: {str(e)}")
        else:
            print(f"No valid subtitle pairs found for {drama_name}")
    
    # Create master CSV file with all dramas
    if all_subtitle_pairs:
        master_df = pd.DataFrame(all_subtitle_pairs)
        master_file = os.path.join(output_dir, 'master_aligned_subtitles.csv')
        try:
            if os.path.exists(master_file):
                existing_master_df = pd.read_csv(master_file)
                master_df = pd.concat([existing_master_df, master_df], ignore_index=True)
                master_df = master_df.drop_duplicates(subset=['Drama Name', 'Episode No#', 'URL', 'Sentence No', 'Urdu Sentence', 'English Sentence'])
            master_df.to_csv(master_file, index=False, encoding='utf-8')
            print(f"\nSuccessfully updated master file {master_file} with {len(master_df)} total subtitle pairs")
            
            # Print statistics
            drama_stats = master_df.groupby('Drama Name').size()
            print("\nSubtitle pairs per drama:")
            for drama, count in drama_stats.items():
                print(f"{drama}: {count} pairs")
            
        except Exception as e:
            print(f"Error saving master CSV file: {str(e)}")
    else:
        print("\nNo subtitle pairs found in any drama")

if __name__ == "__main__":
    main() 