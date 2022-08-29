import pandas as pd

# import domestic + international data
df_2021 = pd.read_csv('/Users/lenni/PycharmProjects/Candidate_Airports/US data/Logan airport (last 6 months)/T_T100_SEGMENT_US_CARRIER_ONLY_2021.csv')
df_2022 = pd.read_csv('/Users/lenni/PycharmProjects/Candidate_Airports/US data/Logan airport (last 6 months)/T_T100_SEGMENT_US_CARRIER_ONLY_2022.csv')

df = pd.concat([df_2021, df_2022])

# Inputs
airport = 'Logan'
avg_speed = 500  # Rough average commercial passenger jet flight speed (m/h)
time_thesh = 3  # Filter by flights less the time_thresh (hours)

# Candidate airport codes;
#   - identification number assigned by US DOT to identify a unique airport (more stable then 3-digit alphanumeric code)
logan_code = 10721

# Init lookup tables
l_airline_id = pd.read_csv('US data/Lookup tables/L_AIRLINE_ID.csv')
l_airline_id.rename(columns={'Code': 'AIRLINE_ID', 'Description': 'CARRIER_NAME'}, inplace=True)
l_airline_id['CARRIER_NAME'] = l_airline_id['CARRIER_NAME'].str.split(': ').str[0]

l_airport_id = pd.read_csv('US data/Lookup tables/L_AIRPORT_ID.csv')

def preprocess(df):
    #  1) remove flights not carrying any passengers (e.g. cargo planes)

    # Keep only passenger carrying planes: service classes A, C, E, F, L (see lookup table)
    df_temp = df.loc[df['CLASS'].isin(['A', 'C', 'E', 'F', 'L'])]

    # Keep only Jet engine planes: aircraft group 6,7,8 (see lookup table)
    # df_temp = df_temp.loc[df['AIRCRAFT_GROUP'].isin([6, 7, 8])]

    # This doesn't catch all cargo planes or planes without passengers for whatever reason
    # Define screen by passengers/departure. If this is <10, then it's very likely not a passenger plane
    df_temp = df_temp.loc[df_temp['DEPARTURES_PERFORMED'] != 0]  # drop anomalies
    df_temp['PASS/DEP'] = df_temp['PASSENGERS'] / df_temp['DEPARTURES_PERFORMED']
    df_temp = df_temp.loc[df_temp['PASS/DEP'] > 10]

    # 2) add useful columns
    df_temp['APPROX_TIME'] = df_temp['DISTANCE'] / avg_speed

    return df_temp


df1 = preprocess(df)


# Get all arrival flights to an airport
def get_arrivals(df, code):
    return df.loc[df['DEST_AIRPORT_ID'] == code]


df_logan = get_arrivals(df1, logan_code)

# select last 6 months with available data
range_2021 = [3,4,5,6,7,8,9,10,11,12]

df_logan_2021 = df_logan.loc[df_logan['YEAR']==2021]
df_logan_2022 = df_logan.loc[df_logan['YEAR']==2022]

df_logan_2021_inrange = df_logan_2021.loc[df_logan_2021['MONTH'].isin(range_2021)]

df_logan1 = pd.concat([df_logan_2022,df_logan_2021_inrange])

df_logan1.to_csv('Logan_arrivals_mar2021-feb2022.csv')




