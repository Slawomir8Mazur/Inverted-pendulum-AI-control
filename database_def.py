from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

import numpy as np
import pandas as pd


class Record:
    def __init__(self, *args, **kwargs):
        for key, item in kwargs.items():
            setattr(self, key, item)


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