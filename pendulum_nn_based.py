from database_def import Record

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
db_folder_name = 'main_databases'
os.chdir(os.getcwd() + '//' + db_folder_name)
names = os.listdir()


def munge_records(names_table, order=1, target_features=['__x', '_x', 'x', '__fi', '_fi', 'fi'], drop_some=False):
    """
    Example usage:
    input, output = munge_records(names[10:13], drop_some=(50, 100), order=3)
    :param names_table: list with names of files with databases
    :param order: timespane between samples
    :param target_features: ['__x', 'x', 'fi']
    :param drop_some: False, (50, 100) drops 50 from the beggining and 100 from the end of EACH frame
    :return: for default features returns two np.arrays,
    input_table with shape <L, 8> and
    target table with shape <L, 6>
    """
    input_table = []
    target_table = []
    r = Record()

    for name in names_table:
        # Loading data
        while name[-3:] == '.db':
            name = name[:-3]
        r.load_from_database(database_name=name)

        # Dropping to much data
        if drop_some:
            r.last_movement.drop(r.last_movement.index[:drop_some[0]], inplace=True)
            r.last_movement.drop(r.last_movement.index[-drop_some[1]:], inplace=True)
            r.last_movement.reset_index(drop=True, inplace=True)

        # Munging data
        dt = r.last_movement.loc[2, 't'] - r.last_movement.loc[1, 't']
        r.last_movement['t_nn'] = dt*order
        r.last_movement['force_nn'] = int(name.split('_')[1])
        input_features = target_features.copy()
        [input_features.append(feature) for feature in ['t_nn', 'force_nn']]

        # Normalise fi
        r.last_movement['fi'] = r.last_movement['fi'] % (2 * np.pi)

        for r_index in r.last_movement.index[:-order]:
            input_table.append(r.last_movement.loc[r_index, input_features])
            target_table.append(r.last_movement.loc[r_index+order, target_features])

    input_table = np.array(input_table)
    target_table = np.array(target_table)

    return input_table, target_table


train_input, train_output = munge_records(names[::3],
                                          drop_some=(1, 700), order=5)
test_input, test_output = munge_records(names[1::33],
                                          drop_some=(1, 950), order=5)

output_size = 6
model = keras.Sequential([
    keras.layers.Dense(output_size+2),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(output_size)
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

model.fit(train_input, train_output, epochs=15)
test_loss = model.evaluate(test_input, test_output)
#print('test accuracy', test_acc, end='\n\n')
print('test loss', test_loss, end='\n')
model.save_weights('512r_256r.h5')