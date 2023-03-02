import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import domestic + international data
df_2021 = pd.read_csv('/Users/lenni/PycharmProjects/Candidate_Airports/US data/Logan airport (last 6 months)/T_T100_SEGMENT_US_CARRIER_ONLY_2021.csv')
df_2022 = pd.read_csv('/Users/lenni/PycharmProjects/Candidate_Airports/US data/Logan airport (last 6 months)/T_T100_SEGMENT_US_CARRIER_ONLY_2022.csv')

df = pd.concat([df_2021, df_2022])

# Inputs
airport = 'Logan'
avg_speed = 500  # Rough average commercial passenger jet flight speed (m/h)
time_thesh = 0  # Filter by flights less the time_thresh (hours)

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

# select 12 months with available data
range_2021 = [3,4,5,6,7,8,9,10,11,12]

df_logan_2021 = df_logan.loc[df_logan['YEAR']==2021]
df_logan_2022 = df_logan.loc[df_logan['YEAR']==2022]

df_logan_2021_inrange = df_logan_2021.loc[df_logan_2021['MONTH'].isin(range_2021)]

df_logan1 = pd.concat([df_logan_2022,df_logan_2021_inrange])

df_logan1.to_csv('Logan_arrivals_mar2021-feb2022.csv')

# Analysis 1: Flights and passenger by flight time

n_flights = []
n_passengers = []
labels = []

for b in np.arange(0,16,1):
    temp = df_logan1.loc[(df_logan1['APPROX_TIME'] >= b) & (df_logan1['APPROX_TIME'] < b+1)]
    n_flights.append(temp['DEPARTURES_PERFORMED'].sum())
    n_passengers.append(temp['PASSENGERS'].sum())
    labels.append('{}'.format(b+1))


X_axis = np.arange(len(labels))

plt.bar(X_axis + 0.2, n_passengers, 0.4, color='C0', label='Passengers')
plt.xticks(X_axis, labels)
plt.xlabel("Flight time")
plt.ylabel("Number of passengers")
plt.title('Arriving passengers by flight time at BOS between Mar 2021 and Feb 2022', y=1.07)
plt.show()

plt.bar(X_axis - 0.2, n_flights, 0.4, color='C1', label='Flights')
plt.xticks(X_axis, labels)
plt.xlabel("Flight time")
plt.ylabel("Number of flights")
plt.title('Arriving flights by flight time at BOS between Mar 2021 and Feb 2022')
plt.show()

# Analysis 2: Flights and passengers per month

n_flights = []
n_passengers = []
labels = []

for month in np.arange(1,13,1):
    temp = df_logan1.loc[df_logan1['MONTH'] == month]
    n_flights.append(temp['DEPARTURES_PERFORMED'].sum())
    n_passengers.append(temp['PASSENGERS'].sum())
    labels.append(month)

X_axis = np.arange(len(labels))

plt.bar(X_axis + 0.2, n_passengers, 0.4, color='C0', label='Passengers')
plt.xticks(X_axis, labels)
plt.xlabel("Month")
plt.ylabel("Number of passengers")
plt.title('Arriving passengers per month at BOS between Mar 2021 and Feb 2022', y=1.07)
plt.show()

plt.bar(X_axis - 0.2, n_flights, 0.4, color='C1', label='Flights')
plt.xticks(X_axis, labels)
plt.xlabel("Month")
plt.ylabel("Number of flights")
plt.title('Arriving flights per month at BOS between Mar 2021 and Feb 2022')
plt.show()

# Analysis 3: Origin country of all flights

n_flights = []
n_passengers = []
labels = []

for country in df_logan1['ORIGIN_COUNTRY_NAME'].unique():
    temp = df_logan1.loc[df_logan1['ORIGIN_COUNTRY_NAME'] == country]
    n_flights.append(temp['DEPARTURES_PERFORMED'].sum())
    n_passengers.append(temp['PASSENGERS'].sum())
    labels.append(country)

df_country_p = pd.DataFrame({'Country':labels, 'Passengers': n_passengers}).sort_values(by=['Passengers'])
df_country_f = pd.DataFrame({'Country':labels, 'Flights': n_flights}).sort_values(by=['Flights'])

Y_axis = np.arange(len(labels))

fig, axes = plt.subplots(figsize=(11,9))
plt.subplots_adjust(left=0.19)
plt.barh(Y_axis, df_country_p['Passengers'], 0.8, color='C0', label='Passengers')
plt.yticks(Y_axis, df_country_p['Country'])
plt.xscale('log')
plt.xlabel('Number of passengers')
plt.ylabel('')
plt.title('Arriving passengers by origin country at BOS between Mar 2021 and Feb 2022 (all flights)', y=1.07)
plt.show()

fig, axes = plt.subplots(figsize=(11,9))
plt.subplots_adjust(left=0.19)
plt.barh(Y_axis, df_country_f['Flights'], 0.8, color='C1', label='Flights')
plt.yticks(Y_axis, df_country_f['Country'])
plt.xscale('log')
plt.xlabel('Number of flights')
plt.ylabel('')
plt.title('Arriving flights by origin country at BOS between Mar 2021 and Feb 2022 (all flights)', y=1.07)
plt.show()

# Analysis 3: Origin country of flights > 3 hours

n_flights = []
n_passengers = []
labels = []

def filter_by_flight_time(df, thresh):
    return df.loc[df['APPROX_TIME'] > thresh]

df_logan1_filt = filter_by_flight_time(df_logan1, 3)

for country in df_logan1['ORIGIN_COUNTRY_NAME'].unique():
    temp = df_logan1_filt.loc[df_logan1_filt['ORIGIN_COUNTRY_NAME'] == country]
    n_flights.append(temp['DEPARTURES_PERFORMED'].sum())
    n_passengers.append(temp['PASSENGERS'].sum())
    labels.append(country)

df_country_p = pd.DataFrame({'Country':labels, 'Passengers': n_passengers}).sort_values(by=['Passengers'])
df_country_f = pd.DataFrame({'Country':labels, 'Flights': n_flights}).sort_values(by=['Flights'])

Y_axis = np.arange(len(labels))

fig, axes = plt.subplots(figsize=(11,9))
plt.subplots_adjust(left=0.19)
plt.barh(Y_axis, df_country_p['Passengers'], 0.8, color='C0', label='Passengers')
plt.yticks(Y_axis, df_country_p['Country'])
plt.xscale('log')
plt.xlabel('Number of passengers')
plt.ylabel('')
plt.title('Arriving passengers by origin country at BOS between Mar 2021 and Feb 2022 (flights > 3 hr)', y=1.07)
plt.show()

fig, axes = plt.subplots(figsize=(11,9))
plt.subplots_adjust(left=0.19)
plt.barh(Y_axis, df_country_f['Flights'], 0.8, color='C1', label='Flights')
plt.yticks(Y_axis, df_country_f['Country'])
plt.xscale('log')
plt.xlabel('Number of flights')
plt.ylabel('')
plt.title('Arriving flights by origin country at BOS between Mar 2021 and Feb 2022 (flights > 3 hr)', y=1.07)
plt.show()

