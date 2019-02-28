from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

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
            self.record = pd.DataFrame(columns=['M_1', 'I_2', 'L',
                                                'A_1', 'V_1', 'U_1',
                                                'E_2', 'W_2', 'Fi_2',
                                                'K', 'B'],
                                       index=[1],
                                       dtype=np.float32)

        for key, value in kwargs.items():
            self.record[key] = value


engine = create_engine('sqlite:///parameters.db', echo=True)
Base = declarative_base()


class RecordBase(Base):
    """
    Class containing parameters of pendulum at one moment.
    All those parameters will be saved into database "parameters.db"
    """
    def __init__(self, *args, **kwargs):
        for key, item in kwargs.items():
            setattr(self, key, item)

    __tablename__ = 'record'
    id = Column(Integer, primary_key=True)  #holds unique id of sample

    ''' mass constants'''
    M_1 = Column(Float)
    I_2 = Column(Float)
    L = Column(Float)
    ''' kinematics variables'''
    A_1 = Column(Float)
    V_1 = Column(Float)
    U_1 = Column(Float)
    E_2 = Column(Float)
    W_2 = Column(Float)
    Fi_2 = Column(Float)
    ''' variables of reaction'''
    K = Column(Float)
    B = Column(Float)


Base.metadata.create_all(engine)