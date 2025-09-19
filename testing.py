import pandas as pd

def analyze_drop_off_spots(file_path):
    """
    Analyzes the top drop-off locations for a specific age group and day of the week.

    Args:
        file_path (str): The path to the CSV data file.
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # Filter the DataFrame for trips by 18-24 year-olds
        # The 'Time' column is converted to datetime for easier comparison
        filtered_df = df[
            (df['Age'] >= 18) &
            (df['Age'] <= 24) &
            (df['Day of Week'] == 'Monday') &
            (df['Time'].str.slice(0, 2).astype(int) >= 20) &
            (df['Time'].str.slice(0, 2).astype(int) < 24)
        ]

        # Count the occurrences of each drop-off address
        top_locations = filtered_df['Drop Off Address'].value_counts()

        # Display the top locations
        if not top_locations.empty:
            print("Top drop-off spots for 18-24 year-olds on Sunday nights:")
            print(top_locations.head())
        else:
            print("No data found for the specified criteria.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except KeyError as e:
        print(f"Error: Missing column {e} in the dataset.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Replace 'your_file_name.csv' with the actual file name
analyze_drop_off_spots('processed_merged_data_with_day.csv')


# import pandas as pd

# def analyze_large_group_trips(file_path):
#     """
#     Analyzes when large groups typically ride downtown.

#     Args:
#         file_path (str): The path to the CSV data file.
#     """
#     try:
#         # Load the dataset
#         df = pd.read_csv(file_path)

#         # Define keywords for downtown destinations
#         downtown_keywords = ['downtown', '6th street', 'west 6th st', 'e 6th st', 'austin', '78701']

#         # Filter the DataFrame for large groups and downtown destinations
#         downtown_trips = df[df['Total Passengers'] >= 6].copy()
        
#         # Check if the 'Drop Off Address' column exists and filter on it
#         if 'Drop Off Address' in downtown_trips.columns:
#             downtown_trips = downtown_trips[
#                 downtown_trips['Drop Off Address'].str.contains('|'.join(downtown_keywords), case=False, na=False)
#             ]

#         if not downtown_trips.empty:
#             # Analyze by Day of Week
#             day_counts = downtown_trips['Day of Week'].value_counts()
#             most_popular_day = day_counts.idxmax()
            
#             # Convert 'Time' to a datetime object to extract the hour
#             downtown_trips['Hour'] = pd.to_datetime(downtown_trips['Time']).dt.hour
#             hour_counts = downtown_trips['Hour'].value_counts()
#             most_popular_hour = hour_counts.idxmax()
            
#             print("Analysis of large group trips (6+ riders) to downtown:")
#             print(f"The most popular day is: {most_popular_day}")
#             print(f"The most popular hour is: {most_popular_hour}:00")
#             print("\nBreakdown by day of the week:")
#             print(day_counts)
#             print("\nBreakdown by hour:")
#             print(hour_counts.sort_index())
#         else:
#             print("No large group trips to downtown found in the dataset.")

#     except FileNotFoundError:
#         print(f"Error: The file '{file_path}' was not found.")
#     except KeyError as e:
#         print(f"Error: Missing column {e} in the dataset.")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

# # Replace 'your_file_name.csv' with the actual file name
# analyze_large_group_trips('processed_merged_data_with_day.csv')