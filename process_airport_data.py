import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np

# Inputs
airport = 'TF Green'

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


def airport_flight_hist(dfi, dfd, format='time'):
    df = pd.concat([dfi, dfd])  # concats international and domestic

    distances = []
    for i, row in df.iterrows():
        distances.append(int(row['DEPARTURES_PERFORMED']) * [row['DISTANCE']])

    distances = list(itertools.chain(*distances))

    if format == 'distance':
        plt.hist(distances, bins=20)
        plt.title('Histogram of flight distances at {}'.format(airport))
        plt.xlabel('Flight Distance (miles)')
        plt.ylabel('Number of flights')
        plt.show()
    elif format == 'time':
        avg_speed = 500  # Rough average commercial passenger jet flight speed (m/h)
        time = np.array(distances) / avg_speed  # time in hours

        plt.hist(time, bins=20)
        plt.title('Histogram of flight approx. time at {}'.format(airport))
        plt.xlabel('Approx. flight time (hours)')
        plt.ylabel('Number of flights')
        plt.show()
    else:
        return "format must be {'distance','time'}"


airport_flight_hist(dfi_loc, dfd_loc, format='time')


def airline_flight_hist(dfi, dfd, format='time'):
    df = pd.concat([dfi, dfd])  # concats international and domestic
    carriers = df.groupby('AIRLINE_ID')
    carriers_sum = carriers.sum()

    tot_pass = carriers_sum['PASSENGERS'].astype(int).sort_values(ascending=False)

    df_top5 = pd.DataFrame(tot_pass).head(5)  # Get top 5 airlines by passenger
    df_top5 = df_top5.merge(l_airline_id, on='AIRLINE_ID')  # merge with airline name

    fig, axes = plt.subplots(5, 1, sharex='all', sharey='all', figsize=(8,15), constrained_layout=True)

    a = 0
    for i, row in df_top5.iterrows():
        id = row['AIRLINE_ID']
        name = row['CARRIER_NAME']
        airline = carriers.get_group(id)

        distances = []
        for ii, row2 in airline.iterrows():
            distances.append(int(row2['DEPARTURES_PERFORMED']) * [row2['DISTANCE']])

        distances = list(itertools.chain(*distances))

        if format == 'distance':
            axes[a].hist(distances, bins=15, orientation='horizontal', facecolor='C{}'.format(a))
            axes[a].set_ylabel(name)
        elif format == 'time':
            avg_speed = 500  # Rough average commercial passenger jet flight speed (m/h)
            time = np.array(distances) / avg_speed  # time in hours

            axes[a].hist(time, bins=15, orientation='horizontal', facecolor='C{}'.format(a))
            axes[a].set_ylabel(name)
        else:
            return "format must be {'distance','time'}"

        a += 1

    old_ylim = axes[0].get_ylim()
    if old_ylim[1] > 10:
        axes[0].set_ylim(old_ylim[0],10.0)

    plt.xlabel('Number of flights')
    fig.supylabel('Flight time (hours)')
    fig.suptitle('Flight time histograms for top 5 airlines at {}'.format(airport))
    plt.show()

airline_flight_hist(dfi_loc, dfd_loc)
