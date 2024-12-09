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
    co_ed_sports,
    HOUSE_COLORS,
    HOUSE_NAMES,
)
import random
import os
import math

all_user_info = {}

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

def allocate_members_to_houses(members_dict, individual_sport_ordering, df):
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
        "Co-Ed": {**co_ed_sports}
    }

    # Track available slots per sport per house
    available_slots = {
        house: {
            gender: {sport: capacity[1] * capacity[0] for sport, capacity in capacities[gender].items()}
            for gender in ["Male", "Female", "Co-Ed"]
        }
        for house in HOUSE_COLORS
    }

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
    
    # Assignment process
    # Priority ordering lists
    sport_allocation_ordering = {
        "Co-Ed": ['Cricket'],
        "Male": ['Football', 'Basketball', 'Volleyball (boys)', 'Ultimate Frisbee'],
        "Female": ['Football', 'Ultimate Frisbee', 'Basketball', 'Throwball (girls)']
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
        "Co-Ed": {**co_ed_sports}
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
                
    return team_allocations

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


def write_allocations_to_excel(team_allocations):
    output_file = os.path.join(os.getcwd(), "allocations.xlsx")
    sport_gender_map = {}
    COLUMN_ORDER = [
    "House", "Sport", "Team Number", "Email", "First Name", "Last Name", 
    "Youth Festival ID", "Gender", "WhatsApp Mobile Number (including country code)", "Age"
    ]

    for house, gender_map in team_allocations.items():
        for gender, sports_map in gender_map.items():
            for sport, teams in sports_map.items():
                rows = []
                for team_number, members in teams.items():
                    for member in members:
                        rows.append({
                            "House": house,
                            "Gender": gender,
                            "Sport": sport,
                            "Team Number": team_number,
                            "First Name": member.get("First Name"),
                            "Last Name": member.get("Last Name"),
                            "Youth Festival ID": member.get("Youth Festival ID"),
                            "WhatsApp Mobile Number (including country code)": member.get("WhatsApp Number"),
                            "Gender": member.get("Gender"),
                            "Email": member.get("Email"),
                            "Age": int(member.get("Age")) if not math.isnan(member.get("Age", float('nan'))) else 0
                        })
                    rows.append({key: "" for key in COLUMN_ORDER})

                    # Convert rows to a DataFrame
                if((gender, sport) not in sport_gender_map):
                    sport_gender_map[(gender, sport)] = []
                sport_gender_map[(gender, sport)].extend(rows)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sport, gender in sport_gender_map:
            rows = sport_gender_map[(sport, gender)]
            df = pd.DataFrame(rows, dtype=str)
            df = df[COLUMN_ORDER]

            # Write to a sheet named after the sport
            sheet_name = f"{sport[:28]} ({gender})"  # Sheet names limited to 31 chars
            df.to_excel(writer, sheet_name=sheet_name, index=False)


    print(f"Team allocations written to {output_file}")

def write_individual_allocations_to_excel(individual_allocations):
    """
    Writes individual sport allocations to an Excel file.

    Parameters:
        individual_allocations (dict): A dictionary containing individual sport allocations.
        all_user_info (dict): A dictionary mapping YFID to user data.
    """
    output_file = os.path.join(os.getcwd(), "individual_allocations.xlsx")
    COLUMN_ORDER = [
        "House", "Sport", "Email", "First Name", "Last Name",
        "Youth Festival ID", "Gender", "WhatsApp Mobile Number (including country code)", "Age"
    ]

    sport_gender_map = {}

    # Process individual allocations
    for sport, house_map in individual_allocations.items():
        for house, gender_map in house_map.items():
            for gender, yfids in gender_map.items():
                rows = []
                for yfid in yfids:
                    user_data = all_user_info.get(yfid, {})
                    rows.append({
                        "House": house,
                        "Sport": sport,
                        "First Name": user_data.get("First Name", ""),
                        "Last Name": user_data.get("Last Name", ""),
                        "Youth Festival ID": yfid,
                        "Gender": user_data.get("Gender", ""),
                        "WhatsApp Mobile Number (including country code)": user_data.get("WhatsApp Number", ""),
                        "Email": user_data.get("Email", ""),
                        "Age": int(user_data.get("Age")) if not math.isnan(user_data.get("Age", float('nan'))) else 0
                    })

                rows.append({key: "" for key in COLUMN_ORDER})  # Add a blank row between groups

                # Append rows to the sport-gender map
                if (gender, sport) not in sport_gender_map:
                    sport_gender_map[(gender, sport)] = []
                sport_gender_map[(gender, sport)].extend(rows)

    # Write to Excel
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for (gender, sport), rows in sport_gender_map.items():
            df = pd.DataFrame(rows, dtype=str)
            df = df[COLUMN_ORDER]  # Ensure column order matches the specified order

            # Create a sheet name with sport and gender
            sheet_name = f"{sport[:28]} ({gender})"  # Limit sheet names to 31 characters
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Individual allocations written to {output_file}")


input_file = "Cleaned Data All Participants.xlsx"
data = pd.read_excel(input_file, skiprows=4)
house_mapping = {'Spearheads': 'Green', 'Pioneers': 'Red', 'Trailblzrs': 'Yellow', 'Mavericks': 'Blue'}
data['House Allotment'] = data['House Allotment'].replace(house_mapping)

sports_by_gender, individual_sport_gender_dict, individual_sport_ordering = (organize_data_by_gender_and_sport(data))
#print(len(sports_by_gender['Male']['Cricket']) + len(sports_by_gender['Female']['Cricket']))
houses = allocate_members_to_houses(sports_by_gender, individual_sport_ordering, data)
house_gender_sport_map = create_house_gender_sport_map(sports_by_gender, houses)
teams = allocate_to_teams(house_gender_sport_map)
individual_teams = allocate_individuals_to_sports(individual_sport_ordering, houses)
write_allocations_to_excel(teams)
write_individual_allocations_to_excel(individual_teams)
