import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np

# Inputs
airport = 'Logan'

# Candidate airport codes;
#   - identification number assigned by US DOT to identify a unique airport (more stable then 3-digit alphanumeric code)
ap_codes = {
    'Nassau': 13605,
    'Bradley': 10529,
    'TF Green': 14307,
    'Logan': 10721,
    'Birmingham': 10608
}

ap_c = ap_codes[airport]

# Init data
df_international = pd.read_csv('/Users/lenni/PycharmProjects/Candidate_Airports/T_T100I_SEGMENT_ALL_CARRIER.csv')
df_domestic = pd.read_csv('/Users/lenni/PycharmProjects/Candidate_Airports/T_T100D_SEGMENT_ALL_CARRIER.csv')

# Init lookup tables
l_airline_id = pd.read_csv('/Users/lenni/PycharmProjects/Candidate_Airports/Lookup tables/L_AIRLINE_ID.csv')
l_airline_id.rename(columns={'Code': 'AIRLINE_ID', 'Description': 'CARRIER_NAME'}, inplace=True)
l_airline_id['CARRIER_NAME'] = l_airline_id['CARRIER_NAME'].str.split(': ').str[0]

l_airport_id = pd.read_csv('/Users/lenni/PycharmProjects/Candidate_Airports/Lookup tables/L_AIRPORT_ID.csv')


# TODO: as a sanity check, compare data before and after preprocessing for each airport


def preprocess(df):
    #  remove flights not carrying any passengers (e.g. cargo planes)

    # Keep only passenger carrying planes: service classes A, C, E, F, L (see lookup table)
    df_temp = df.loc[df['CLASS'].isin(['A', 'C', 'E', 'F', 'L'])]

    # Keep only Jet engine planes: aircraft group 6,7,8 (see lookup table)
    df_temp = df_temp.loc[df['AIRCRAFT_GROUP'].isin([6, 7, 8])]

    # This doesn't catch all cargo planes or planes without passengers for whatever reason
    # Define screen by passengers/departure. If this is <10, then it's very likely not a passenger plane
    df_temp = df_temp.loc[df_temp['DEPARTURES_PERFORMED'] != 0]  # drop anomalies
    df_temp['PASS/DEP'] = df_temp['PASSENGERS'] / df_temp['DEPARTURES_PERFORMED']
    df_temp = df_temp.loc[df_temp['PASS/DEP'] > 10]

    return df_temp


dfi = preprocess(df_international)
dfd = preprocess(df_domestic)


# Get all arrival flights to an airport
def get_arrivals(df, code):
    return df.loc[df['DEST_AIRPORT_ID'] == code]


def get_departures(df, code):
    return df.loc[df['ORIGIN_AIRPORT_ID'] == code]


dfi_loc = get_arrivals(dfi, ap_c)
dfd_loc = get_arrivals(dfd, ap_c)


def airport_summary_stats(dfi, dfd):
    print('2019 Summary Statistics for {}\n---------------------------------'.format(airport))

    A = int(dfi['DEPARTURES_PERFORMED'].sum())
    B = int(dfd['DEPARTURES_PERFORMED'].sum())
    C = A + B

    D = int(dfi['PASSENGERS'].sum())
    E = int(dfd['PASSENGERS'].sum())
    F = D + E

    print('Number of arriving international passenger planes: ', A)
    print('Number of arriving domestic passenger planes:', B)
    print('Total number of arriving planes:', C)

    print('---------------------------------')

    print('Number of arriving international passengers:', D)
    print('Number of arriving domestic passengers:', E)
    print('Total number of arriving passengers:', F)

    # A. Number of arriving international passenger planes
    # B. Number of arriving domestic passenger planes
    # C. Total number of arriving planes

    # D. Number of arriving international passengers
    # E. Number of arriving domestic passengers
    # F. Total number of arriving passengers


airport_summary_stats(dfi_loc, dfd_loc)


def top_airlines(dfi, dfd, by='PASSENGERS'):
    # by: {PASSENGERS, DEPARTURES_PERFORMED}
    # d/b/a abbreviation means Doing Business As.

    # Group by carrier ID
    carriers = pd.concat([dfi, dfd]).groupby('AIRLINE_ID')  # concats international and domestic
    intl_carriers = dfi.groupby('AIRLINE_ID')
    dom_carriers = dfd.groupby('AIRLINE_ID')

    # Sum columns over airline groups (do this once to save resources_
    carriers_sum = carriers.sum()
    intl_carriers_sum = intl_carriers.sum()
    dom_carriers_sum = dom_carriers.sum()

    # Get total number of arriving passengers by airline
    tot_pass = carriers_sum['PASSENGERS'].astype(int).sort_values(
        ascending=False)  # Irrespective of int'l vs. domestic status
    tot_intl_pass = intl_carriers_sum['PASSENGERS'].astype(int).sort_values(ascending=False)
    tot_dom_pass = dom_carriers_sum['PASSENGERS'].astype(int).sort_values(ascending=False)

    # Get total number of arriving flights by airline
    tot_arrivals = carriers_sum['DEPARTURES_PERFORMED'].astype(int).sort_values(
        ascending=False)  # Irrespective of int'l vs. domestic status
    tot_intl_arrivals = intl_carriers_sum['DEPARTURES_PERFORMED'].astype(int).sort_values(ascending=False)
    tot_dom_arrivals = dom_carriers_sum['DEPARTURES_PERFORMED'].astype(int).sort_values(ascending=False)

    if by == 'PASSENGERS':
        # Get top 20 airlines by passenger number
        df_top20 = pd.DataFrame(tot_pass).head(20)
        df_intl_top20 = pd.DataFrame(tot_intl_pass).head(20)
        df_dom_top20 = pd.DataFrame(tot_dom_pass).head(20)

        # Merge with number of arrivals data
        df_top20 = df_top20.merge(tot_arrivals, on='AIRLINE_ID')
        df_intl_top20 = df_intl_top20.merge(tot_intl_arrivals, on='AIRLINE_ID')
        df_dom_top20 = df_dom_top20.merge(tot_dom_arrivals, on='AIRLINE_ID')

    elif by == 'DEPARTURES_PERFORMED':
        # Get top 20 airlines by arrivals
        df_top20 = pd.DataFrame(tot_arrivals).head(20)
        df_intl_top20 = pd.DataFrame(tot_intl_arrivals).head(20)
        df_dom_top20 = pd.DataFrame(tot_dom_arrivals).head(20)

        # Merge with number of passengers data
        df_top20 = df_top20.merge(tot_pass, on='AIRLINE_ID')
        df_intl_top20 = df_intl_top20.merge(tot_intl_pass, on='AIRLINE_ID')
        df_dom_top20 = df_dom_top20.merge(tot_dom_pass, on='AIRLINE_ID')
    else:
        return "'by' must be set to 'PASSENGER' or 'DEPARTURES_PERFORMED'"

    # Add Airline name column
    df_top20 = df_top20.merge(l_airline_id, on='AIRLINE_ID')
    df_intl_top20 = df_intl_top20.merge(l_airline_id, on='AIRLINE_ID')
    df_dom_top20 = df_dom_top20.merge(l_airline_id, on='AIRLINE_ID')

    # TODO add distribution statistics (e.g. median) about time of flight and/or distance travelled

    return df_top20, df_intl_top20, df_dom_top20


df_top20, df_intl_top20, df_dom_top20 = top_airlines(dfi_loc, dfd_loc)


def flight_dist_hist(dfi, dfd, format='distance'):
    df = pd.concat([dfi, dfd])  # concats international and domestic

    distances = []
    for i, row in df.iterrows():
        distances.append(int(row['DEPARTURES_PERFORMED']) * [row['DISTANCE']])

    distances = list(itertools.chain(*distances))

    if format == 'distance':
        plt.hist(distances,bins=50)
        plt.title('Histogram of flight distances at {}'.format(airport))
        plt.xlabel('Flight Distance (miles)')
        plt.ylabel('Count')
        plt.show()
    elif format == 'time':
        avd_speed = 500  # Rough average commercial passenger jet flight speed (m/h)
        time = np.array(distances)/avd_speed # time in hours

        plt.hist(time, bins=50)
        plt.title('Histogram of flight approx. time at {}'.format(airport))
        plt.xlabel('Approx. flight time (hours)')
        plt.ylabel('Count')
        plt.show()
    else:
        return "format must be {'distance','time'}"

d = flight_dist_hist(dfi_loc, dfd_loc,format='time')

# TODO:
#       2. Histogram of distances for arriving planes
#       2. Most common origin airports/locations
