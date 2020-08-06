import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

#PATH = "C:/Users/Administrator/Desktop/14days/EDA/New York Airbnb Data Exploration/"
# Load data
nyc_airbnb = pd.read_csv("AB_NYC_2019.csv")

# Data Describe
nyc_airbnb.shape
nyc_airbnb.info()
nyc_airbnb.describe

# Check Missing Value
nyc_airbnb.isnull().any()

# pairplot
n_columns = ['price', 'number_of_reviews', 'availability_365', 'minimum_nights']
fig = sns.pairplot(nyc_airbnb[n_columns], plot_kws={"s": 8})
plt.tight_layout()
plt.show(fig)

# The First question!!
# Which hosts are the busiest and why?
plt.figure(figsize=(30, 5))
fig_bar = sns.countplot(x=nyc_airbnb['host_name'], linewidth=3)
fig_bar.set_xticklabels(fig_bar.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=5)
plt.tight_layout()
plt.show(fig_bar)
# because this plot too big , let hosts owner more than 100 rooms , then get busiest_hosts
host_name = nyc_airbnb['host_name'].unique()
busiest_hosts = []
for name in host_name:
    if nyc_airbnb.loc[nyc_airbnb['host_name'] == name, :].shape[0] > 100:
        busiest_hosts.append(name)
busiest_hosts_df = nyc_airbnb.loc[nyc_airbnb['host_name'].isin(busiest_hosts)]  # get busiest_hosts_df
# plot countplot
plt.figure(figsize=(20, 5))
fig_bar = sns.countplot(x=busiest_hosts_df['host_name'], linewidth=3)
fig_bar.set_xticklabels(fig_bar.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=8)
plt.tight_layout()
plt.show(fig_bar)

# get hosts_group and hosts_areas  to comfirm who's house across many areas
hosts_group = busiest_hosts_df.groupby(['host_name', 'neighbourhood_group']).size()
hosts_areas = busiest_hosts_df.groupby(['host_name', 'neighbourhood_group']).size().groupby('host_name').size()
busiest_hosts_df.groupby(['host_name', 'neighbourhood_group']).size().groupby('neighbourhood_group').size()


def find_busiest_hosts(busiest_hosts_df, hosts_areas, busiest_hosts, busiest_hosts_answer):
    #  number of room more than 300~500  and  across 5 areas
    # because all of busiest_hosts is number of room more than 100, so we can search more than 100
    for number_of_room in range(300, 600, 100):
        for name in busiest_hosts:
            if busiest_hosts_df.loc[busiest_hosts_df['host_name'] == name].shape[0] >= number_of_room and hosts_areas[
                name] >= 5:
                busiest_hosts_answer.append(name)

    # Because range is 300~500, so have duplicate hosts. It's have to duplicate.
    busiest_hosts_answer = list(dict.fromkeys(busiest_hosts_answer))
    # Compare which hosts have most house.
    if busiest_hosts_df.loc[busiest_hosts_df['host_name'] == busiest_hosts_answer[0]].shape[0] > \
            busiest_hosts_df.loc[busiest_hosts_df['host_name'] == busiest_hosts_answer[1]].shape[0]:
        return (busiest_hosts_answer[0])
    else:
        return (busiest_hosts_answer[0])


busiest_hosts_answer = []
print("The first question answer, busiest hosts is",
      find_busiest_hosts(busiest_hosts_df, hosts_areas, busiest_hosts, busiest_hosts_answer))

# The second question!!
# What areas have more traffic than others and why is that the case?
# now this step will discuss relationship between neighbourhood_group and price, number_of_reviews etc...
group = nyc_airbnb.groupby(
    "neighbourhood_group")  # 跟據neighbourhood_gruop分組  有Bronx, Brooklyn, Manhattan, Queens, Staten Island
# 透過group.size() 我們可以觀察到   在Brooklyn 跟 Manhattan的房子數量很多
group_Bronx = group.get_group('Bronx')
fig_density = sns.kdeplot(group_Bronx['price'])
plt.title("Bronx price per room")
plt.xlabel('price')
plt.ylabel('density')
plt.show(fig_density)

# Bronx  price per room
plt.figure(figsize=(30, 5))
fig_bar = sns.countplot(x=group_Bronx['price'], linewidth=3)
fig_bar.set_xticklabels(fig_bar.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=5)
plt.tight_layout()
plt.show(fig_bar)

# Brooklyn price per room
group_Brooklyn = group.get_group('Brooklyn')
plt.figure(figsize=(30, 5))
fig_bar = sns.countplot(x=group_Brooklyn['price'], linewidth=3)
fig_bar.set_xticklabels(fig_bar.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=5)
plt.tight_layout()
plt.show(fig_bar)
# Manhattan price per room
group_Manhattan = group.get_group('Manhattan')
plt.figure(figsize=(30, 5))
fig_bar = sns.countplot(x=group_Manhattan['price'], linewidth=3)
fig_bar.set_xticklabels(fig_bar.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=5)
plt.tight_layout()
plt.show(fig_bar)
# Queens price per room
group_Queens = group.get_group('Queens')
plt.figure(figsize=(30, 5))
fig_bar = sns.countplot(x=group_Queens['price'], linewidth=3)
fig_bar.set_xticklabels(fig_bar.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=5)
plt.tight_layout()
plt.show(fig_bar)
# Staten Island price per room
group_Staten_Island = group.get_group('Staten Island')
plt.figure(figsize=(30, 5))
fig_bar = sns.countplot(x=group_Staten_Island['price'], linewidth=3)
fig_bar.set_xticklabels(fig_bar.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=5)
plt.tight_layout()
plt.show(fig_bar)

# The Third Question
# Are there any relationships between prices, number of reviews, and the number of days that a given listing is booked?
# Use scatter plot
# Observed price and number of reviews
plt.figure(figsize=(30, 5))
scatter_plot = sns.scatterplot(x='price', y='number_of_reviews', data=nyc_airbnb, s=10)
plt.show(scatter_plot)

plt.figure(figsize=(30, 5))
scatter_plot = sns.scatterplot(x='price', y='number_of_reviews', data=nyc_airbnb, hue='neighbourhood_group', s=10)
plt.show(scatter_plot)
# we found the lower price will get more reviews. Conversely,the higer price will get less reviews.
# Because the higer price of apartment give better rental quality, and customer think it should be, so customer may do not review.
# Now we add number of days
n_columns = ['price', 'number_of_reviews', 'availability_365']
fig = sns.pairplot(nyc_airbnb[n_columns], corner=True, plot_kws={"s": 8}, height=3, aspect=1)
plt.tight_layout()
plt.show(fig)
