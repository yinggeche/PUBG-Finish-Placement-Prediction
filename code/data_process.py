import numpy as np
import pandas as pd

def featureModify():
    print("Loading data from data.csv")
    all_data = pd.read_csv('../input/data.csv')
    # Cleaning the data by filtering the maxPlace > 1
    all_data = all_data[all_data['maxPlace'] > 1]
    # Cleaning the data by deleting the winPlacePerc = 0
    all_data = all_data[all_data['winPlacePerc'].notnull()]

    # Cleaning the data by converting string-type feature to number-type feature
    all_data['matchType'] = all_data['matchType'].map({
    'crashfpp':1,
    'crashtpp':2,
    'duo':3,
    'duo-fpp':4,
    'flarefpp':5,
    'flaretpp':6,
    'normal-duo':7,
    'normal-duo-fpp':8,
    'normal-solo':9,
    'normal-solo-fpp':10,
    'normal-squad':11,
    'normal-squad-fpp':12,
    'solo':13,
    'solo-fpp':14,
    'squad':15,
    'squad-fpp':16
    })

    # Adding features by lifting
    print("Adding match size feature...")
    matchSizeData = all_data.groupby(['matchId']).size().reset_index(name='matchSize')
    all_data = pd.merge(all_data, matchSizeData, how='left', on=['matchId'])
    del matchSizeData

    # Adding other features by lifting
    all_data['_walkDistance_kills_Ratio'] = all_data['walkDistance'] / all_data['kills']
    all_data['_totalDistance'] = 0.25*all_data['rideDistance'] + all_data["walkDistance"] + all_data["swimDistance"]
    all_data['_totalDistancePerDuration'] =  all_data["_totalDistance"]/all_data["matchDuration"]
    all_data['killPlacePerc'] = all_data.groupby('matchId')['killPlace'].rank(pct=True).values
    all_data['killPerc'] = all_data.groupby('matchId')['kills'].rank(pct=True).values
    all_data['walkDistancePerc'] = all_data.groupby('matchId')['walkDistance'].rank(pct=True).values
    all_data['_kill_kills_Ratio2'] = all_data['killPerc']/all_data['walkDistancePerc']
    all_data['_killPlace_walkDistance_Ratio2'] = all_data['walkDistancePerc']/all_data['killPlacePerc']
    all_data = all_data.drop(["walkDistancePerc","killPerc","_totalDistance","assists","boosts","swimDistance","rideDistance","walkDistance","killStreaks","longestKill","matchDuration","weaponsAcquired","DBNOs","numGroups","vehicleDestroys","rankPoints","revives"], axis = 1)
    # Adding group size feature
    print("Adding group size...")
    groupSize = all_data.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    all_data = pd.merge(all_data, groupSize, how='left', on=['matchId', 'groupId'])
    del groupSize
    # Rearrange the order of features
    y = all_data['Id','winPlacePerc']
    all_data = all_data.drop(["winPlacePerc"], axis = 1)
    all_data = pd.merge(all_data, y, how='left', on=['Id'])
    # Define the infinit
    all_data[all_data == np.Inf] = np.NaN
    all_data[all_data == np.NINF] = np.NaN
    all_data.fillna(0, inplace=True)

    # Drop the 'Id' column
    all_data = all_data.drop(columns='Id')
    return all_data


# Split the data by dividing them into train set and test set using input fraction
def split_train_val(data, fraction):
    matchIds = data['matchId'].unique().reshape([-1])
    train_size = int(len(matchIds)*fraction)

    random_idx = np.random.RandomState(seed=2).permutation(len(matchIds))
    train_matchIds = matchIds[random_idx[:train_size]]
    val_matchIds = matchIds[random_idx[train_size:]]

    data_train = data.loc[data['matchId'].isin(train_matchIds)]
    data_val = data.loc[data['matchId'].isin(val_matchIds)]
    return data_train, data_val

# Find the average statics of each group
def ave_statis(data):
    mean_data = all_data.groupby('groupId').mean()
    mean_data = mean_data/100000
    arry = np.array(all_data)
    meanarr = np.array(mean_data)
    list = meanarr.tolist()
    return list

def write_data(data, path):
    with open(path, 'a') as f:
        for x in data:
            str3 = ','.join([str(y) for y in x])
            f.write(str3)
            f.write('\n')

all_data = featureModify()
print("Split the data into train set and test set")
train_data, test_data = split_train_val(all_data, 0.5)
train_data.to_csv("train_data.csv", index=True)
list1 = ave_statis(train_data)
list2 = ave_statis(test_data)
write_data(list1,'train.data')
write_data(list2,'test.data')
print("Write the data into text file")
