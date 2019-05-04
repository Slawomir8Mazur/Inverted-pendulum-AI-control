from database_def import Record
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import InputLayer, Dense
import matplotlib.pyplot as plt


class DynamicLearning:
    def __init__(self, **kwargs):
        self.model = Sequential()
        self.model.add(InputLayer(batch_input_shape=(1, 6)))
        self.model.add(Dense(12, activation='relu'))
        self.model.add(Dense(3, activation='linear'))
        self.model.compile(loss='mse',
                           optimizer='adam',
                           metrics=['mae'])
        self.param = {
            'y': 0.1,
            'eps': 0.5,
            'decay': 0.995
        }

        for key in kwargs:
            self.param[key] = kwargs[key]

        self.ft_table = self.ft_define()

    def teach_greedy(self, num_episodes=100, range_of_angle=(1, 360), epochs=1, pass_to_reward={}):
        feats = ['__x', '_x', 'x', '__fi', '_fi', 'fi']
        for i in range(num_episodes):
            j = 0
            record = Record()
            record.position_set(np.random.randint(*range_of_angle))
            self.param["eps"] *= self.param["decay"]
            print("Episode {} of {}".format(i, num_episodes))
            for j in range(15):
                target = []
                for a in range(0, 3):
                    next_record = Record(record)
                    next_record.move(force_table=[(self.ft_table.loc[a, 'forces'],
                                                   self.ft_table.loc[a, 'times'])])
                    #target.append(self.give_reward(next_record.record[feats], **pass_to_reward)['fi'][0])        # Constant move of trolley
                    target.append(self.give_reward(next_record.record[feats], **pass_to_reward)[['fi', 'x']].sum(axis=1)[0])


                self.model.fit(record.record[feats], np.array(target).reshape(-1, 3), epochs=epochs, verbose=0)
                a = np.argmax(self.model.predict(record.record[feats]))
                record.move(force_table=[(self.ft_table.loc[a, 'forces'],
                                          self.ft_table.loc[a, 'times'])])
                print("From:\t %0.2f \tTo:\t %0.2f" % (self.give_reward(record.record[feats], **pass_to_reward)[['fi', 'x']].sum(axis=1)[0],
                                                       self.give_reward(record.last_movement.iloc[[0], :][feats], **pass_to_reward)[['fi', 'x']].sum(axis=1)[0]))


    def teach(self, num_episodes=100, range_of_angle=(1, 360), epochs=1, pass_to_reward={}):
        #r_avg_list = []
        feats = ['__x', '_x', 'x', '__fi', '_fi', 'fi']
        for i in range(num_episodes):
            record = Record()
            record.position_set(np.random.randint(*range_of_angle))
            self.param["eps"] *= self.param["decay"]
            print("Episode {} of {}".format(i, num_episodes))
            #r_sum = 0
            while self.give_reward(record.record[feats], **pass_to_reward)['fi'][0] < 14.9:
                rec_old = record.record[feats].copy()
                if np.random.random() < self.param['eps']:
                    a = np.random.randint(0, 3)
                else:
                    a = np.argmax(self.model.predict(rec_old))

                record.move(force_table=[(self.ft_table.loc[a, 'forces'],
                                          self.ft_table.loc[a, 'times'])])
                target = (self.give_reward(record.record[feats], **pass_to_reward)[['fi', 'x']].sum(axis=1)[0]
                          + self.param['y'] * np.max(self.model.predict(record.record[feats])))
                target_vec = self.model.predict(rec_old)[0]
                target_vec[a] = target
                self.model.fit(rec_old, target_vec.reshape(-1, 3), epochs=epochs, verbose=0)
                #r_sum += self.give_reward(record.record[feats], **pass_to_reward)
            #r_avg_list.append(r_sum / 1000)
        #return r_avg_list

    # Check if that function works
    def control(self, record, num_episodes=100):
        feats = ['__x', '_x', 'x', '__fi', '_fi', 'fi']
        for i in range(num_episodes):
            a = np.argmax(self.model.predict(record.record[feats]).reshape(1, -1))
            record.move(force_table=[(self.ft_table.loc[a, 'forces'],
                                      self.ft_table.loc[a, 'times'])])
        return record

    def check(self, angle=20, num_episodes=20):
        r = Record()
        r.position_set(angle)
        self.control(r, num_episodes)
        r.visualize(['move'], r.stack_of_movement, separately=True)

    def ft_define(self, **kwargs):
        ft_param = {
            "forces": [500, 0, -500],
            "times": [0.1]
        }
        for key in kwargs:
            self.param[key] = kwargs[key]

        df_values, list_of_keys = self.iter_through_dict(ft_param)

        table_ft = pd.DataFrame(df_values, columns=list_of_keys[::-1])
        return table_ft

    @staticmethod
    def iter_through_dict(dic):
        list_of_lists = []
        master_list = []
        list_of_keys = []

        [(list_of_lists.append(dic[key]), list_of_keys.append(key))
         for key in dic]

        def iter_through(master_list, list_of_lists, *args):
            if len(list_of_lists) == 0:
                master_list.append(args)
            else:
                for elem in list_of_lists[0]:
                    iter_through(master_list, list_of_lists[1:], elem, *args)

        iter_through(master_list, list_of_lists)
        return master_list, list_of_keys

    @staticmethod
    def give_reward(record, **kwargs):
        param = {
            'fi_mul': 0.5,
            '_fi_mul': 5,
            '__fi_mul': 1,
            '_fi_max': 30,
            '__fi_max': 100,
            'x_mul': -0.2
        }
        for key in kwargs:
            param[key] = kwargs[key]

        reward = pd.DataFrame()

        reward['fi'] = param['fi_mul'] * np.power(np.abs((record['fi'] % (2 * np.pi) - np.pi)), 3)
        reward['_fi'] = param['_fi_mul'] * np.abs(np.abs(record['_fi']) - param['_fi_max']) - param['_fi_max'] * param['_fi_mul']
        reward['__fi'] = param['__fi_mul'] * np.abs(np.abs(record['__fi']) - param['__fi_max']) - param['__fi_max'] * param['__fi_mul']
        reward['x'] = param['x_mul'] * np.abs(record['x'])

        reward['total'] = reward['__fi'] + reward['_fi'] + reward['fi'] + reward['x'] + 50

        return reward

"""
d = DynamicLearning(eps=0.1)
d.model.load_weights('reinforced_learning_greedy_lin_epoch20_4.h5')
d.check(num_episodes=500)
plt.show()
"""

d = DynamicLearning(eps=0.1)
d.model.load_weights('reinforced_learning_greedy_lin_epoch20_with_epoch_param_1.h5')
d.check(num_episodes=80)
for epoch in [5, 4, 3, 2, 1]:
    d.teach_greedy(num_episodes=50, epochs=epoch)
d.model.save_weights('reinforced_learning_greedy_lin_epoch20_with_epoch_param_2.h5')
d.check(num_episodes=80)
plt.show()


"""r = Record()
r.position_set(20)
r.move([(2000, 0.2)])
r.visualize(['move'], r.last_movement, separately=True)
print(DynamicLearning().give_reward(r.last_movement))
plt.show()"""

# temp2 = pd.concat([temp.stack_of_movement, temp.last_movement, temp.record], ignore_index=True, sort=False)
