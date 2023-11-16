import unittest
import pickle
import pandas as pd
import numpy as np
import random as rd
from pyomo.environ import *

modelo_1 = pickle.load(open('modelo_regresion/modelo_1.pkl','rb'))


class TestSum(unittest.TestCase):
    def test_estatico_input_ml(self):
        """
        Se realiza el test para el valor de la restriccion 1 con un conjunto aleatorio de pesos pequeños
        """
        data = pd.DataFrame({'Años': [30],'Años_de_experiencia': [4], 'Cargo': ['Data Engineer']})


        sueldo=int(modelo_1.predict(data)[0])

        self.assertGreaterEqual(sueldo,0)

    def test_dinamico_input_ml(self):
        """
        Se realiza el test para el valor de la restriccion 1 con un conjunto aleatorio de pesos pequeños
        """
        edad=rd.randint(0,100)
        exp=rd.randint(0,100)
        cargo='Data Engineer'

        data = pd.DataFrame({'Años': [edad],'Años_de_experiencia': [exp], 'Cargo': [cargo]})


        sueldo=int(modelo_1.predict(data)[0])

        self.assertGreaterEqual(sueldo,0)

    def test_dinamico_high_bound_input_ml(self):
        """
        Se realiza el test para el valor de la restriccion 1 con un conjunto aleatorio de pesos pequeños
        """
        edad=rd.randint(1000000,10000000)
        exp=rd.randint(1000000,10000000)
        cargo='Data Engineer'

        data = pd.DataFrame({'Años': [edad],'Años_de_experiencia': [exp], 'Cargo': [cargo]})


        sueldo=int(modelo_1.predict(data)[0])

        self.assertGreaterEqual(sueldo,0)
    
if __name__ == '__main__':
    unittest.main()