### Custom definitions and classes if any ###
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

def predictRuns(testInput):
    #read data
    testInput = pd.read_csv(testInput)
    data = pd.read_csv('all_matches.csv')
    #filter useful data
    data = data.tail(15296)
    data = data[['batting_team', 'bowling_team', 'venue','innings', 'ball', 'runs_off_bat']]
    data = data[data['ball'] < 6]
    
    #transform data to our comfort
    inning = 1
    mod_data = []
    sum_run = 0
    bat = ''
    bowl = ''
    ven = ''
    for d in data.values:
        if d[3] != 1 and d[3] != 2:
            continue
        if d[3] != inning:
            mod_data.append([bat, bowl, ven, sum_run])
            sum_run = 0
            inning = d[3]
        else:
            sum_run += d[5]
            bat = d[0]
            bowl = d[1]
            ven = d[2]

    mod_data.append([bat, bowl, ven, sum_run])
    
    #calc no. of columns in test_data
    bat_count = data['batting_team'].unique()
    bat_count = len(bat_count)
    bowl_count = data['bowling_team'].unique()
    bowl_count = len(bowl_count)
    venue_count = data['venue'].unique()
    venue_count = len(venue_count)
    team_count = bat_count + bowl_count + venue_count
    
    #naming columns of transformed and selected data
    m_data = pd.DataFrame(mod_data, columns = ['batting_team', 'bowling_team', 'venue', 'runs'])
    
    #split data into feature and target
    X = m_data.iloc[:, :-1]
    y = m_data.iloc[:, -1]
    
    #convert strings to numbers
    X = pd.get_dummies(X)
    var = X.columns
    
    #train the model(fit the data)
    forest_model = RandomForestRegressor(random_state=0)
    forest_model.fit(X.values, y.values)
    
    #convert test data to match with our model's input format
    X_test = []
    for i in range(0, len(var)):
        X_test.append(0)
    X_test = np.array([X_test])
    X_test.reshape(len(var),1)
    X_test = pd.DataFrame(X_test, columns = var)
    X_test['batting_team_' + testInput['batting_team'].values[0]] = 1
    X_test['bowling_team_'+ testInput['bowling_team'].values[0]] = 1
    
    #make predictions on the input data
    y_pred = forest_model.predict(X_test)

    #little intution based on observations
    tot_batsmen_played = len(testInput['batsmen'].values[0].split(','))
    tot_batsmen_played -= 2
    tot_batsmen_played *= 3
    y_pred -= tot_batsmen_played
    if tot_batsmen_played == 0:
        y_pred += 5
    #making prediction into integer
    prediction = round(y_pred[0])
    return prediction
