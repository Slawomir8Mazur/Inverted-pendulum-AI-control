'''from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
'''
import numpy as np
import pandas as pd


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
        return pd.DataFrame(columns=['M_1', 'I_2', 'L',
                                     'A_1', 'V_1', 'U_1',
                                     'E_2', 'W_2', 'Fi_2',
                                     'K', 'B'],
                            index=[0],
                            dtype=np.float32)

    @staticmethod
    def new_movement_record():
        return pd.DataFrame(columns=['A_1', 'V_1', 'U_1',
                                     'E_2', 'W_2', 'Fi_2'],
                            index=[1],
                            dtype=np.float32)

    def give_movement_param(self):
        column_names = ['A_1', 'V_1', 'U_1', 'E_2', 'W_2', 'Fi_2']
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
        input_table = [10, 100, 1,
                       0, 0, 0,
                       0, 0, angle*np.pi/180,
                       0, 0]
        self.record.astype(np.float32)
        for pos, key in enumerate(self.record.columns):
            self.record[key] = input_table[pos]

    def update_stacks(self):
        self.stack_of_movement = self.stack_of_movement.append(self.last_movement, ignore_index=True)
        self.update_last_movement()
        #self.last_movement = self.record

    def update_last_movement(self):
        self.last_movement = self.last_movement.append(self.give_movement_param(), ignore_index=True)

    def single_move(self, force_table_record):
        self.update_last_movement()
        new_record = self.record.copy()
        force, dt = force_table_record

        """ movement equations"""
        F_1 = self.record['M_1']*9.81*np.sin(self.record['Fi_2'])*np.cos(self.record['Fi_2']) + force
        M_2 = self.record['L']*self.record['M_1']*9.81*np.sin(self.record['Fi_2'])/2 \
              - self.record['L']*np.cos(self.record['Fi_2'])*force/2

        new_record['A_1'] = F_1 / new_record['M_1']
        new_record['E_2'] = M_2 / new_record['I_2']

        new_record['V_1'] = self.record['V_1'] + new_record['A_1']/dt
        new_record['W_2'] = self.record['W_2'] + new_record['E_2'] / dt

        new_record['U_1'] = self.record['U_1'] + new_record['V_1'] / dt
        new_record['Fi_2'] = self.record['Fi_2'] + new_record['W_2'] / dt

        self.record = new_record
        return new_record

    def move(self, force_table, dt_min=0.02):
        self.update_stacks()

        for F, t in force_table:
            if t > dt_min:
                dt = int(t//dt_min)
                for i in range(dt):
                    self.single_move((F, dt))
                self.single_move((F, t%dt))
            else:
                self.single_move((F, t))

    def visealize(self, features, source, separately):
        if features == 'all':
            features = self.new_record().columns
            pass


r = Record()
r.position_set(30)
print(r.give_movement_param())
r.move([(0, 0.1)], dt_min=0.01)
print(r.last_movement.to_string())
print(r.stack_of_movement.to_string(), end='\n-----------------------')
