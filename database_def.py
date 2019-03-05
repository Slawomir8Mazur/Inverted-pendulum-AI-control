'''from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as snb


class Record:
    """
    Holds values of state of pendulum
    example initiation:

    record = Record(previous_record, M_1=11.2, L=280)

    , where previous_record is of instance Record
    """
    def __init__(self, *args, **kwargs):
        if args:
            if isinstance(args[0], Record):
                self.record = args[0]
        else:
            self.record = self.new_record()

        for key, value in kwargs.items():
            self.record[key] = value
        
        self.stack_of_movement = self.new_movement_record()
        self.last_movement = self.new_movement_record()

    @staticmethod
    def new_record():
        return pd.DataFrame(columns=['m_1', 'm_2', 'l', 
                                     '__x', '_x', 'x',
                                     '__fi', '_fi', 'fi',
                                     'g', 't'],
                            index=[0],
                            dtype=np.float32)

    @staticmethod
    def new_movement_record():
        return pd.DataFrame(columns=['__x', '_x', 'x',
                                     '__fi', '_fi', 'fi',
                                     't'],
                            index=[1],
                            dtype=np.float32)

    def give_movement_param(self):
        column_names = self.new_movement_record().columns
        return self.record[column_names]

    def give_angle(self):
        """ Operates with degrees"""
        return self.record['Fi_2']/np.pi*180

    def change_angle(self, angle_value):
        """ Operates with degrees"""
        self.record['Fi_2'] = np.pi*angle_value/180

    def dummy_set(self):
        for key in self.record.columns:
            self.record[key] = np.random.random()

    def position_set(self, angle):
        input_table = [20, 2, 1, 
                       0, 0, 0,
                       0, 0, angle*np.pi/180.0,
                       9.81, 0]
        for pos, key in enumerate(self.record.columns):
            self.record[key] = input_table[pos]

    def update_stacks(self):
        self.stack_of_movement = self.stack_of_movement.append(self.last_movement, 
                                                               ignore_index=True)
        self.update_last_movement()

    def update_last_movement(self):
        self.last_movement = self.last_movement.append(self.give_movement_param(), ignore_index=True)

    def single_move(self, force_table_record):
        self.update_last_movement()
        new_record = self.record.copy()
        force, dt = force_table_record

        """ movement equations"""

        new_record['__x'] = (- np.multiply.reduce([self.record['m_2'],
                                                   self.record['l'],
                                                   np.power(self.record['_fi'], 2),
                                                   np.sin(self.record['fi'])])
                             + np.multiply.reduce([self.record['m_2'],
                                                   self.record['l'],
                                                   self.record['__fi'],
                                                   np.cos(self.record['fi'])])
                             + force
                             ) / np.add(self.record['m_1'],
                                        self.record['m_2'])

        new_record['__fi'] = (+ np.multiply.reduce([self.record['g'],
                                                   np.sin(self.record['fi'])])
                             + np.multiply.reduce([new_record['__x'],
                                                   np.cos(self.record['fi'])])
                             ) / self.record['l']

        new_record['_x'] = self.record['_x'] + new_record['__x']*dt
        new_record['_fi'] = self.record['_fi'] + new_record['__fi']*dt

        new_record['x'] = self.record['x'] + new_record['_x']*dt
        new_record['fi'] = self.record['fi'] + new_record['_fi']*dt
        new_record['t'] = self.record['t'] + dt

        self.record = new_record
        return new_record

    def move(self, force_table, dt_min=0.02):
        self.update_stacks()

        for F, t in force_table:
            if t > dt_min:
                rng = int(t//dt_min)
                for i in range(rng):
                    self.single_move((F, dt_min))
                self.single_move((F, t%dt_min))
            else:
                self.single_move((F, t))

    def visualize(self, features, source, separately, stop=False):
        """
        Function plotting pointed features of source DataFrame
        :param features: list of features, ex. ['A_1', 'E_2']
        :param source: where from print
        :param separately: True if you dont want to print on the same plot
        :param stop: If True computation is stopped after showing plot, if False ALLWAYS USE plt.show() later
        """
        if features == ['all']:
            features = self.new_record().columns
        if features == ['move']:
            features = self.new_movement_record().columns

        ''' Look for errors'''
        if 'fi' in features:
            new_col_name = 'fi[deg]'
            features = [elem if elem != 'fi' else new_col_name for elem in features]
            source[new_col_name] = source['fi'] / np.pi * 180

        if 't' in features:
            features = [feature for feature in features if feature is not 't']

        if separately:
            for feature in features:
                plt.figure()
                plt.plot(source['t'], source[feature])
                plt.title(feature)
                plt.ylabel(feature)
                plt.xlabel('time[s]')
        else:
            f_len = len(features)
            plt.figure()
            for i, feature in enumerate(features):
                plt.subplot(1, f_len, i+1)
                plt.plot(source['t'], source[feature])
                plt.title(feature)
                plt.ylabel(feature)
                plt.xlabel('time[s]')

        if not stop:
            plt.draw()
        else:
            plt.show()

        def save_to_database(self, what='all', if_exists='replace'):
            engine = create_engine('sqlite:///parameters.db')
            if what=="all":
                what = [self.record,
                        self.last_movement,
                        self.stack_of_movement]
            for source in what:
                source.to_sql(source.__name___,
                              engine,
                              if_exists=if_exists)

        def load_from_database(self, where_to='all'):
            engine = create_engine('sqlite:///parameters.db')
            if what=='all':
                what = [self.record,
                        self.last_movement,
                        self.stack_of_movement]
            for source in where_to:
                source = pd.read_sql_table(source.__name__,
                                           con=engine)

if __name__ == '__main__':
    r = Record()
    r.position_set(130)
    r.move([(10, 5), ], dt_min=0.05)
    r.visualize(['move'], r.last_movement, False)
    plt.show()
else:
    print('It is not database.py file')
    r = Record()
    r.position_set(130)
    r.move([(10, 5), ], dt_min=0.05)
       

"""
Next to do:
check saving to and loading from database

"""
