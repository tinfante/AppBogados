#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from random import shuffle
import sys
from time import sleep
import fasttext


respuestas = {
        'abuso_sexual': 'Creo que me estás describiendo un caso de abuso sexual.',
        'acoso_laboral': 'Creo que me estás describiendo un caso de acoso laboral',
        'despido_injustificado': 'Creo que me estás describiendo unc aso de despido injustificado'
        }

def load():
    print('\nCargando modelo', end='')
    model = fasttext.load_model('model/model.bin')
    for i in range(5):
        sleep(.2)
        print('.', end='')
        sys.stdout.flush()
    print('\n')
    return model


def load_tests():
    with open('data/test.txt', 'r') as f:
        test_cases = [l.strip().split(' ', 1) for l in f.read().decode('utf-8').split('\n') if l.strip()]
    shuffle(test_cases)
    return test_cases


def test(model, test_cases):
    print('Evaluando con %d casos. ' % len(test_cases), end='')
    correct = 0
    for l, t in test_cases:
        sleep(.2)
        if l == model.predict(t)[0][0]:
            print('✓', end='')
            correct += 1
        else:
            print('x', end='')
        sys.stdout.flush()
    sleep(.2)
    print('\nAccuracy: %.2f' % (correct / float(len(test_cases))))


def main():
    model = load()
   
    test_cases = load_tests()
    test(model, test_cases)

    print()
    while True:
        print(u'AppBoard: ¡Hola! Cuéntamos tu problema.')
        print('Usuario: ', end='')
        text = raw_input().decode('utf-8')
        pred = model.predict(text)[0][0][9:]
        print(respuestas[pred], '\n')


if __name__ == '__main__':
    main()
