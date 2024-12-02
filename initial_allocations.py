import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sports_constants import (
    group_a_boys_capacities,
    group_a_girls_capacities,
    group_b_boys_capacities,
    group_b_girls_capacities,
    co_ed_sports,
    HOUSE_COLORS,
    HOUSE_NAMES,
)
import random
import os

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
    
    # Iterate through the rows of the DataFrame
    for _, row in df.iterrows():
        # Extract gender
        gender = row["Gender:"]
        
        # Extract user info
        user_info = {
            "Email": row["Email Address"],
            "First Name": row["First Name:"],
            "Last Name": row["Last Name:"],
            "Youth Festival ID": row["Youth Festival ID:"],
            "Gender": gender,
            "WhatsApp Number": row["WhatsApp Mobile Number (including country code):"],
            "Priority": int(row["priority"])
        }
        
        # Loop through group preferences to sort by sport
        for col in ["Group A preference", "Group B preference"]:
            sport = row[col]
            if sport != "NONE of the above" and pd.notna(sport) and sport.strip():
                gender_dict[gender][sport].append(user_info)
    
    return gender_dict

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

def allocate_members_to_houses(members_dict):
    # Initialize house allocations
    house_allocations = {}

    # Combine capacities for ease of access
    capacities = {
        "Male": {**group_a_boys_capacities, **group_b_boys_capacities},
        "Female": {**group_a_girls_capacities, **group_b_girls_capacities},
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

    # Shuffle house order for fair assignment
    house_order = HOUSE_COLORS[:]
    random.shuffle(house_order)

    # Assignment process
    # member_dict = {gender: sport: [users_who_want_to_play]}
    male_sport_allocation_ordering = ['Football', 'Basketball', 'Volleyball (boys)', 'Ultimate Frisbee']
    female_sport_allocation_ordering = ['Football', 'Ultimate Frisbee', 'Basketball', 'Throwball (girls)']
    co_ed_sports_allocation_ordering = ['Cricket']

    gender = 'Female'
    for sport in female_sport_allocation_ordering:
        members_in_sport = members_dict[gender][sport]
        sorted_members = sort_members_by_priority(members_in_sport)
        for member in sorted_members:
            if member['Youth Festival ID'] in house_allocations:
                continue
            allocated = False
            tried_alloc = 0
            while not allocated and tried_alloc < 4:
                house = house_order.pop(0)
                slots_left = available_slots[house][gender][sport]
                if slots_left > 0:
                    allocated = True
                    house_allocations[member['Youth Festival ID']] = house
                    available_slots[house][gender][sport] = available_slots[house][gender][sport] - 1
                house_order.append(house)
                tried_alloc += 1
            if not allocated:
                print(f'Unable to allocate member {member} to sport {sport}')

    gender = 'Male'
    for sport in male_sport_allocation_ordering:
        members_in_sport = members_dict[gender][sport]
        sorted_members = sort_members_by_priority(members_in_sport)
        for member in sorted_members:
            if member['Youth Festival ID'] in house_allocations:
                continue
            allocated = False
            tried_alloc = 0
            while not allocated and tried_alloc < 4:
                house = house_order.pop(0)
                slots_left = available_slots[house][gender][sport]
                if slots_left > 0:
                    allocated = True
                    house_allocations[member['Youth Festival ID']] = house
                    available_slots[house][gender][sport] = available_slots[house][gender][sport] - 1
                house_order.append(house)
                tried_alloc += 1
            if not allocated:
                print(f'Unable to allocate member {member} to sport {sport}')

    gender = 'Co-Ed'
    for sport in co_ed_sports_allocation_ordering:
        members_in_sport = members_dict['Male'][sport] + members_dict['Female'][sport]
        sorted_members = sort_members_by_priority(members_in_sport)
        for member in sorted_members:
            if member['Youth Festival ID'] in house_allocations:
                continue
            allocated = False
            tried_alloc = 0
            while not allocated and tried_alloc < 4:
                house = house_order.pop(0)
                slots_left = available_slots[house][gender][sport]
                if slots_left > 0:
                    allocated = True
                    house_allocations[member['Youth Festival ID']] = house
                    available_slots[house][gender][sport] = available_slots[house][gender][sport] - 1
                house_order.append(house)
                tried_alloc += 1
            if not allocated:
                print(f'Unable to allocate member {member} to sport {sport}')

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

def write_allocations_to_excel(team_allocations):
    output_file = os.path.join(os.getcwd(), "allocations.xlsx")
    sport_gender_map = {}
    COLUMN_ORDER = [
    "House", "Sport", "Team Number", "Email", "First Name", "Last Name", 
    "Youth Festival ID", "Gender", "WhatsApp Mobile Number (including country code)"
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
                            "Email": member.get("Email")
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

input_file = "Cleaned Data All Participants.xlsx"
data = pd.read_excel(input_file, skiprows=2)
# Temp to generate prioirty
data['priority'] = np.random.randint(1, 8, size=len(data))
sports_by_gender = (organize_data_by_gender_and_sport(data))
houses = allocate_members_to_houses(sports_by_gender)
house_gender_sport_map = create_house_gender_sport_map(sports_by_gender, houses)
#print(house_gender_sport_map['Red']['Male']['Volleyball (boys)'])
teams = allocate_to_teams(house_gender_sport_map)
print(teams)
write_allocations_to_excel(teams)
