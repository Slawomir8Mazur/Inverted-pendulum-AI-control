from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

engine = create_engine('sqlite:///parameters.db', echo=True)
Base = declarative_base()

class Record(Base):
    '''
    Class containing parameters of pendulum at one moment.
    All those parameters will be saved into database "parameters.db"
    '''
    __tablename__ = 'record'
    id = Column(Integer, primary_key=True)  #holds unique id of sample

    ''' mass constants'''
    m_1 = Column(Float)
    I_2 = Column(Float)
    L = Column(Float)
    ''' kinematics variables'''
    a_1 = Column(Float)
    v_1 = Column(Float)
    u_1 = Column(Float)
    E_2 = Column(Float)
    w_2 = Column(Float)
    fi_2 = Column(Float)
    ''' variables of reaction'''
    K = Column(Float)
    B = Column(Float)

Base.metadata.create_all(engine)