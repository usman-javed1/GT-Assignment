import pandas as pd

def process_drama_metadata():
    try:
        # Read the Excel file
        df = pd.read_excel('/home/usman/Desktop/GT Assignment /metadata.xlsx')
        
        filtered_df = df[df['Hindi Subtitles File Name'].isna() & (~df['English Subtitles with Timestamp File Name'].isna())]
        
        output_filename = 'drama_metadata_filtered.xlsx'
        filtered_df.to_excel(output_filename, index=False)
        print(f"Successfully processed the file. Output saved to {output_filename}")
        print(f"Removed {len(df) - len(filtered_df)} entries with Hindi subtitles")
        
    except FileNotFoundError:
        print("Error: drama_metadata.xlsx not found in the current directory")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    process_drama_metadata() 