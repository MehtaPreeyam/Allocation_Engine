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

input_file = "Cleaned Data All Participants.xlsx"
data = pd.read_excel(input_file, skiprows=4)
house_mapping = {'Spearheads': 'Green', 'Pioneers': 'Red', 'Trailblzrs': 'Yellow', 'Mavericks': 'Blue'}
data['House Allotment'] = data['House Allotment'].replace(house_mapping)