import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sports_constants import (
    group_a_boys_capacities,
    group_a_girls_capacities,
    group_b_boys_capacities,
    group_b_girls_capacities,
    group_c_boys_sports_capacities,
    group_c_girls_sports_capacities,
    group_d_boys_sports_capacities,
    group_d_girls_sports_capacities,
    HOUSE_COLORS,
    HOUSE_NAMES,
)
import random
import os
import math

all_user_info = {}
marquee_status = defaultdict(list)
yfids_found = set()

def count_members_in_house(houses):
    counter = {}
    for yfid in houses:
        house = houses[yfid]
        if house not in counter:
            counter[house] = 0
        counter[house] += 1
    print(counter)


# Function to create nested dictionaries by gender and sport
def organize_data_by_gender_and_sport(df):
    # Initialize the nested dictionary
    gender_dict = defaultdict(lambda: defaultdict(list))
    individual_sport_gender_dict = defaultdict(lambda: defaultdict(list))
    individual_sport_ordering = defaultdict(list)
    
    # Iterate through the rows of the DataFrame
    for _, row in df.iterrows():
        # Extract gender
        gender = row["Gender:"]
        
        # Extract user info
        user_info = {
            "Email": row["Email Address"],
            "First Name": row["First Name:"],
            "Last Name": row["Last Name:"],
            "Youth Festival ID": row["REG YFID"],
            "Gender": gender,
            "WhatsApp Number": row["WhatsApp Mobile Number (including country code):"],
            "Priority": int(row["Priority"]),
            "Age": row['Age']
        }

        if row['Registered'] == 0 or row['House Allotment'] == False: # skip members who have not been registered properly
            continue
        
        # Loop through group preferences to sort by sport
        for col in ["Group A preference", "Group B preference"]:
            sport = row[col]
            if sport != "NONE of the above" and pd.notna(sport) and sport.strip():
                gender_dict[gender][sport].append(user_info)
        
        group_c_or_d = str(row["Choose Group C or Group D?"])
        yfid = user_info['Youth Festival ID']
        yfids_found.add(yfid) # YFID tracking to see if all have been allocated
        all_user_info[yfid] = user_info
        gender = user_info['Gender']
        priority = user_info['Priority']
        if('Group C' in group_c_or_d):
            for col in ["Group C preference 1", "Group C preference 2", "Group C preference 3"]:
                sport = row[col]
                if sport != "NONE of the above" and pd.notna(sport) and sport.strip():
                    individual_sport_gender_dict[gender][sport].append(user_info)
                    individual_sport_ordering[(yfid, priority, gender)].append(sport)
        elif('Group D' in group_c_or_d):
            for col in ["Group D preference 1", "Group D preference 2"]:
                sport = row[col]
                if sport != "NONE of the above" and pd.notna(sport) and sport.strip():
                    individual_sport_gender_dict[gender][sport].append(user_info)
                    individual_sport_ordering[(yfid, priority, gender)].append(sport)
    
    return gender_dict, individual_sport_gender_dict, individual_sport_ordering

def sort_members_by_priority(member_list):
    # Step 1: Sort by Priority
    sorted_data = sorted(member_list, key=lambda x: x['Priority'])

    # Step 2: Group by Priority
    priority_groups = {}
    for user in sorted_data:
        priority = user['Priority']
        if priority not in priority_groups:
            priority_groups[priority] = []
        priority_groups[priority].append(user)

    # Step 3: Shuffle within each group
    for priority in priority_groups:
        random.shuffle(priority_groups[priority])

    # Step 4: Combine groups into a single list
    final_list = []
    for priority in sorted(priority_groups.keys()):  # Ensure keys are in sorted order
        final_list.extend(priority_groups[priority])
    
    return final_list

def allocate_members_to_houses(members_dict, individual_sport_ordering, df, marquees_df):
    # Initialize house allocations
    house_allocations = {
        row['REG YFID']: row['House Allotment']
        for _, row in df.iterrows()
        if pd.notna(row['House Allotment'])  # Only consider rows with valid House Allotments
    }

    # Combine capacities for ease of access
    capacities = {
        "Male": {**group_a_boys_capacities, **group_b_boys_capacities, **group_c_boys_sports_capacities, **group_d_boys_sports_capacities},
        "Female": {**group_a_girls_capacities, **group_b_girls_capacities, **group_c_girls_sports_capacities, **group_d_girls_sports_capacities},
    }

    # Track available slots per sport per house
    available_slots = {
        house: {
            gender: {sport: capacity[1] * capacity[0] for sport, capacity in capacities[gender].items()}
            for gender in ["Male", "Female"]
        }
        for house in HOUSE_COLORS
    }
    # Initialize marquee tracking: {sport: {gender: {house: count}}}
    marquee_counts = {
        sport: {
            gender: {house: 0 for house in HOUSE_COLORS}
            for gender in ["Male", "Female"]
        }
        for sport in set(marquees_df['2024 Group A Marquee - Sport'].dropna()).union(
            set(marquees_df['2024 Group B Marquee - Sport'].dropna())
        )
    }

    # Update marquee counts for already allocated marquees
    for yfid, house in house_allocations.items():
        marquee_row = marquees_df[marquees_df['REG YFID'] == yfid]
        if marquee_row.empty:
            continue

        gender = marquee_row.iloc[0].get('Gender')
        for col in ['2024 Group A Marquee - Sport', '2024 Group B Marquee - Sport']:
            sport = marquee_row.iloc[0].get(col)
            if pd.notna(sport) and gender in marquee_counts[sport]:
                marquee_counts[sport][gender][house] += 1

    # Filter relevant marquee members
    marquees_df = marquees_df[
        marquees_df['2024 Group A Marquee - Type'].notna() | marquees_df['2024 Group B Marquee - Type'].notna()
    ]

    # Allocate new marquees
    for _, marquee in marquees_df.iterrows():
        yfid = marquee['REG YFID']
        # Extract marquee sports
        marquee_sports = []
        if pd.notna(marquee.get('2024 Group A Marquee - Type')):
            marquee_sports.append(marquee.get('2024 Group A Marquee - Sport'))
            marquee_status[yfid].append((marquee.get('2024 Group A Marquee - Sport'), marquee.get('2024 Group A Marquee - Type')))
        if pd.notna(marquee.get('2024 Group B Marquee - Type')):
            marquee_sports.append(marquee.get('2024 Group B Marquee - Sport'))
            marquee_status[yfid].append((marquee.get('2024 Group B Marquee - Sport'), marquee.get('2024 Group B Marquee - Type')))

        if yfid in house_allocations:
            continue  # Skip if already allocated

        gender = marquee.get('Gender:')
        if gender not in ["Male", "Female"]:
            print(f"Skipping YFID {yfid} due to missing or invalid gender: {gender}")
            continue
        

        # Assign to the house with the fewest marquees for the first sport and gender
        for sport in marquee_sports:
            if sport not in marquee_counts:
                print(f"Sport {sport} not found in marquee_counts. Skipping YFID {yfid}.")
                continue

            # Find the house with the fewest marquee allocations for this sport and gender
            target_house = min(
                HOUSE_COLORS,
                key=lambda house: marquee_counts[sport][gender][house]
            )

            # Allocate to the target house
            house_allocations[yfid] = target_house
            marquee_counts[sport][gender][target_house] += 1
            print(f"Marquee YFID {yfid} allocated to house {target_house} for sport {sport} (Gender: {gender}).")
            break  # Move to the next marquee member after allocation

    for _, row in df.iterrows():
        if pd.notna(row['House Allotment']):
            house = row['House Allotment']
            gender = row['Gender:']
            for sport in capacities[gender]:
                # Reduce available slots based on pre-allocation
                if sport in members_dict[gender]:
                    for member in members_dict[gender][sport]:
                        if member['Youth Festival ID'] == row['Youth Festival ID:']:
                            available_slots[house][gender][sport] -= 1

    # Assignment process
    # Priority ordering lists
    sport_allocation_ordering = {
        "Male": ['Cricket', 'Football', 'Basketball', 'Volleyball (boys)', 'Ultimate Frisbee'],
        "Female": ['Cricket', 'Football', 'Ultimate Frisbee', 'Basketball', 'Throwball (girls)']
    }

    for gender, sports_order in sport_allocation_ordering.items():
        for sport in sports_order:
            members_in_sport = members_dict.get(gender, {}).get(sport, [])
            if gender == "Co-Ed":
                members_in_sport = members_dict['Male'].get(sport, []) + members_dict['Female'].get(sport, [])

            sorted_members = sort_members_by_priority(members_in_sport)
            for member in sorted_members:
                if member['Youth Festival ID'] in house_allocations:
                    continue

                # Allocate to the house with the most slots left for this sport
                target_house = max(
                    HOUSE_COLORS,
                    key=lambda house: available_slots[house][gender][sport]
                )

                # Check if the house has available slots
                if available_slots[target_house][gender][sport] > 0:
                    house_allocations[member['Youth Festival ID']] = target_house
                    available_slots[target_house][gender][sport] -= 1
                else:
                    print(f"Unable to allocate member {member} to sport {sport}")

        # Start Individual house allocs
    sorted_individual_members = sorted(
        individual_sport_ordering.keys(),
        key=lambda x: x[1]  # Sort by priority
    )

    # Iterate through members sorted by priority
    for yfid, prio, member_gender in sorted_individual_members:
        # Skip members already allocated to a house
        if yfid in house_allocations:
            continue

        # Get the list of individual sports for the member in order of preference
        individual_sports = individual_sport_ordering[(yfid, prio, member_gender)]

        # Iterate through the list of sports in preference order
        for sport in individual_sports:
            if sport not in capacities[member_gender]:
                print(f"Sport {sport} not found in capacities for gender {member_gender}. Skipping...")
                continue

            # Find the house with the most slots left for this sport
            target_house = max(
                HOUSE_COLORS,
                key=lambda house: available_slots[house][member_gender].get(sport, 0)
            )

            # Check if the house has available slots for this sport
            if available_slots[target_house][member_gender].get(sport, 0) > 0:
                # Allocate the member to this house
                house_allocations[yfid] = target_house
                # Decrease the available slots for this sport in the target house
                available_slots[target_house][member_gender][sport] -= 1
                print(f"Allocated YFID {yfid} to house {target_house} for sport {sport}.")
                break  # Exit the sports loop once allocated
    
    
    return house_allocations


def create_house_gender_sport_map(sports_by_gender, house_allocations):
    house_gender_sport_map = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list))
    )
    for gender in sports_by_gender:
        sport_map = sports_by_gender[gender]
        for sport in sport_map:
            members_in_sport = sport_map[sport]
            for member in members_in_sport:
                yfid = member['Youth Festival ID']
                if yfid in house_allocations:
                    house = house_allocations[yfid]
                    house_gender_sport_map[house][gender][sport].append(member)
    
    return house_gender_sport_map

                
def allocate_to_teams(house_gender_sport_map):
    # Prepare capacities
    capacities = {
        "Male": {**group_a_boys_capacities, **group_b_boys_capacities},
        "Female": {**group_a_girls_capacities, **group_b_girls_capacities},
    }

    team_allocations = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )  # Nested structure: house -> gender -> sport -> team # -> list of members

    for house, gender_map in house_gender_sport_map.items():
        for gender, sports_map in gender_map.items():
            for sport, members in sports_map.items():
                if sport in capacities[gender]:
                    total_teams, players_per_team = capacities[gender][sport]
                else:
                    print(f"Sport {sport} not found in capacities for gender {gender}. Skipping.")
                    continue
                sorted_members = sort_members_by_priority(members)
                # Initialize team counters
                team_counters = {team: 0 for team in range(1, total_teams + 1)}
                # Round-robin allocation
                for i, member in enumerate(sorted_members):
                    team_number = (i % total_teams) + 1  # Cycle through team numbers
                    if team_counters[team_number] < players_per_team:
                        team_allocations[house][gender][sport][team_number].append(member)
                        team_counters[team_number] += 1
                    else:
                        print(f"Cannot allocate {member} to {sport} (House: {house}, Team: {team_number}) - Capacity Full")
    return team_allocations
"""
    for sport in capacities['Co-Ed']:
        total_teams, players_per_team = capacities["Co-Ed"][sport]
        for house in house_gender_sport_map:
            members = house_gender_sport_map[house]['Female'][sport] + house_gender_sport_map[house]['Male'][sport]
            team_counters = {team: 0 for team in range(1, total_teams + 1)}
            sorted_members = sort_members_by_priority(members)

            for i, member in enumerate(sorted_members):
                team_number = (i % total_teams) + 1  # Cycle through team numbers
                if team_counters[team_number] < players_per_team:
                    team_allocations[house]['Co-Ed'][sport][team_number].append(member)
                    team_counters[team_number] += 1
                else:
                    print(f"Cannot allocate {member} to {sport} (House: {house}, Team: {team_number}) - Capacity Full")
"""             

def allocate_individuals_to_sports(individual_sport_ordering, house_allocations):
    capacities = {
        "Male": {**group_c_boys_sports_capacities, **group_d_boys_sports_capacities},
        "Female": {**group_c_girls_sports_capacities, **group_d_girls_sports_capacities}
    }

    # Initialize allocations: {sport: {house: {gender: []}}}
    sport_allocations = {
        sport: {house: {"Male": [], "Female": []} for house in HOUSE_COLORS}
        for gender in capacities
        for sport in capacities[gender]
    }

    # Initialize the allocations tracker
    available_slots = {
        sport: {
            house: {
                gender: capacities[gender][sport][0] * capacities[gender][sport][1]
                for gender in ["Male", "Female"]
                if sport in capacities[gender]
            }
            for house in HOUSE_COLORS
        }
        for gender in capacities
        for sport in capacities[gender]
    }

    # Sort individual_sport_ordering by priority
    sorted_individual_members = sorted(
        individual_sport_ordering.keys(),
        key=lambda x: x[1]  # Sort by priority
    )

    # Allocate individuals
    for yfid, priority, gender in sorted_individual_members:
        # Skip if the individual is not allocated to a house
        if yfid not in house_allocations:
            continue

        # Get the house for this individual
        house = house_allocations[yfid]

        # Get the individual's preferred sports
        preferred_sports = individual_sport_ordering[(yfid, priority, gender)]

        # Attempt to allocate them to their preferred sport
        for sport in preferred_sports:
            if sport not in capacities[gender]:
                print(f"Sport {sport} not available for gender {gender}. Skipping...")
                continue

            # Check if there's space in this sport for the house and gender
            if available_slots[sport][house][gender] > 0:
                # Allocate the individual to the sport
                sport_allocations[sport][house][gender].append(yfid)
                # Decrease the available slots for this sport in the target house
                available_slots[sport][house][gender] -= 1
                print(f"Allocated YFID {yfid} to sport {sport} in house {house}.")
                break  # Move to the next individual after allocation
            else:
                print(f"No space available for YFID {yfid} in sport {sport} for house {house}.")

    return sport_allocations

def write_combined_allocations_to_excel(team_allocations, individual_allocations):
    output_file = os.path.join(os.getcwd(), "combined_allocations.xlsx")
    sport_gender_map = {}
    COLUMN_ORDER = [
        "House", "Sport", "Team/Player Number", "Email", "First Name", "Last Name", 
        "Youth Festival ID", "Gender", "WhatsApp Mobile Number (including country code)", "Status", "Age"
    ]
    gender_mapping = {'Male': 'boys', 'Female': 'girls'}

    # Capacity dictionaries
    capacities = {
        "Male": {**group_a_boys_capacities, **group_b_boys_capacities, **group_c_boys_sports_capacities, **group_d_boys_sports_capacities},
        "Female": {**group_a_girls_capacities, **group_b_girls_capacities, **group_c_girls_sports_capacities, **group_d_girls_sports_capacities},
    }

    # Process team allocations
    for house, gender_map in team_allocations.items():
        for gender, sports_map in gender_map.items():
            for sport, teams in sports_map.items():
                rows = []
                parsed_gender = gender_mapping[gender]
                sport_with_gender = sport
                if parsed_gender not in sport:
                    sport_with_gender = f'{sport} ({parsed_gender})'

                # Get sport capacity
                total_teams, players_per_team = capacities[gender].get(sport, (0, 0))

                # Process each team
                for team_number in range(1, total_teams + 1):
                    members = teams.get(team_number, [])
                    # Add allocated members
                    for member in members:
                        yfid = member.get("Youth Festival ID")
                        yfids_found.discard(yfid)
                        player_status_tuple = marquee_status.get(yfid, None)
                        player_status = 'Player'
                        if player_status_tuple:
                            for sport_status in player_status_tuple:
                                mq_sport = sport_status[0]
                                if mq_sport == sport:
                                    player_status = sport_status[1]
                                    break

                        rows.append({
                            "House": house,
                            "Gender": gender,
                            "Sport": sport_with_gender,
                            "Team/Player Number": team_number,
                            "First Name": member.get("First Name"),
                            "Last Name": member.get("Last Name"),
                            "Youth Festival ID": member.get("Youth Festival ID"),
                            "WhatsApp Mobile Number (including country code)": member.get("WhatsApp Number"),
                            "Gender": member.get("Gender"),
                            "Email": member.get("Email"),
                            "Age": int(member.get("Age")) if not math.isnan(member.get("Age", float('nan'))) else 0,
                            "Status": player_status
                        })

                    # Add blank rows for unfilled spots
                    remaining_spots = players_per_team - len(members)
                    for _ in range(remaining_spots):
                        rows.append({
                            "House": house,
                            "Gender": gender,
                            "Sport": sport_with_gender,
                            "Team/Player Number": team_number,
                            "First Name": "",
                            "Last Name": "",
                            "Youth Festival ID": "",
                            "WhatsApp Mobile Number (including country code)": "",
                            "Gender": "",
                            "Email": "",
                            "Age": "",
                            "Status": ""
                        })

                    rows.append({key: "" for key in COLUMN_ORDER})  # Blank row between teams

                # Append rows to sport_gender_map
                if (gender, sport) not in sport_gender_map:
                    sport_gender_map[(gender, sport)] = []
                sport_gender_map[(gender, sport)].extend(rows)

    # Process individual allocations
    for sport, house_map in individual_allocations.items():
        for house, gender_map in house_map.items():
            for gender, yfids in gender_map.items():
                rows = []
                player_count = 0
                parsed_gender = gender_mapping[gender]
                sport_with_gender = f'{sport} ({parsed_gender})'

                # Get sport capacity for individual allocations
                total_teams, players_per_team = capacities[gender].get(sport, (0, 0))  # Total slots for the sport

                for yfid in yfids:
                    yfids_found.discard(yfid)
                    user_data = all_user_info.get(yfid, {})
                    player_count += 1
                    rows.append({
                        "House": house,
                        "Sport": sport_with_gender,
                        "Team/Player Number": player_count,
                        "First Name": user_data.get("First Name", ""),
                        "Last Name": user_data.get("Last Name", ""),
                        "Youth Festival ID": yfid,
                        "Gender": user_data.get("Gender", ""),
                        "WhatsApp Mobile Number (including country code)": user_data.get("WhatsApp Number", ""),
                        "Email": user_data.get("Email", ""),
                        "Age": int(user_data.get("Age")) if not math.isnan(user_data.get("Age", float('nan'))) else 0,
                        "Status": ""
                    })

                # Add blank rows for unfilled spots
                remaining_slots = players_per_team - player_count
                for _ in range(remaining_slots):
                    player_count += 1
                    rows.append({
                        "House": house,
                        "Sport": sport_with_gender,
                        "Team/Player Number": player_count,
                        "First Name": "",
                        "Last Name": "",
                        "Youth Festival ID": "",
                        "Gender": "",
                        "WhatsApp Mobile Number (including country code)": "",
                        "Email": "",
                        "Age": "",
                        "Status": ""
                    })

                rows.append({key: "" for key in COLUMN_ORDER})  # Add a blank row between groups

                # Append rows to the sport-gender map
                if (gender, sport) not in sport_gender_map:
                    sport_gender_map[(gender, sport)] = []
                sport_gender_map[(gender, sport)].extend(rows)

    # Write to Excel
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sport, gender in sport_gender_map:
            rows = sport_gender_map[(sport, gender)]
            df = pd.DataFrame(rows, dtype=str)
            df = df[COLUMN_ORDER]

            # Write to a sheet named after the sport
            sheet_name = f"{sport[:28]} ({gender})"  # Sheet names limited to 31 chars
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Combined team and individual allocations written to {output_file}")

input_file = "Cleaned Data All Participants.xlsx"
marquee_file = "marquees.xlsx"
data = pd.read_excel(input_file, skiprows=4)
marquee_df = pd.read_excel(marquee_file, skiprows=3)
joined_df = pd.merge(
        data,
        marquee_df,
        on='Youth Festival ID:',
        how='inner',  # Use 'inner' to include only matching rows
        suffixes=('_regular', '_marquee')  # Add suffixes to distinguish columns
    )

house_mapping = {'Spearheads': 'Green', 'Pioneers': 'Red', 'Trailblzrs': 'Yellow', 'Mavericks': 'Blue'}
data['House Allotment'] = data['House Allotment'].replace(house_mapping)

sports_by_gender, individual_sport_gender_dict, individual_sport_ordering = (organize_data_by_gender_and_sport(data))

houses = allocate_members_to_houses(sports_by_gender, individual_sport_ordering, data, joined_df)
house_gender_sport_map = create_house_gender_sport_map(sports_by_gender, houses)
teams = allocate_to_teams(house_gender_sport_map)
individual_teams = allocate_individuals_to_sports(individual_sport_ordering, houses)
write_combined_allocations_to_excel(teams, individual_teams)
print(yfids_found)