#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8




import unittest
from text_normalize import *


class TestTextNormalization(unittest.TestCase):

    def _testNorm(self, a, b):
        _norm = normalize_text(a, replacements_regex)
        # _norm2 = normalize_text(_norm, replacements_regex)
        if not _norm == b:
            self.fail('\n'+_norm+' <> \n'+b)
        # self.assertEqual(_norm, b)
        # test idempotence
        # self.assertEqual(_norm2, b)

    # def test_de_acronym(self):
    #     # self._testNorm(' стороны, именуемые в дальнейшем совместно «Стороны», а по отдельности - «Сторона», заключили ',
    #     #                ' стороны, заключили ')
    #
    #     # self._testNorm('«ИВа», именуемая в дальнейшем «Исполнитель», ', '«ИВа», именуемое «Исполнитель», ')
    #
    #     self._testNorm('ООО ', 'Общество с ограниченной ответственностью ')
    #     self._testNorm('xx ООО ', 'xx Общество с ограниченной ответственностью ')
    #
    #     self._testNorm('ПАОП', 'ПАОП')
    #     self._testNorm('лиловое АО ', 'лиловое Акционерное Общество ')
    #     self._testNorm('ЗАО ', 'Закрытое Акционерное Общество ')
    #     self._testNorm('XЗАОX', 'XЗАОX')
    #     self._testNorm('витальное ЗАО ', 'витальное Закрытое Акционерное Общество ')
    #



    # #     self._testNorm('смотри п.2.2.2 нау',  'смотри пункт 2.2.2 нау')
    # #     self._testNorm('смотри\n\n п.2.2.2 нау',   'смотри\n пункт 2.2.2 нау')
    #
    # #     self._testNorm(' в п.п. 4.1. – 4.5. ',   ' в пунктах 4.1. – 4.5. ')
    # def test_deacronym_failed(self):
    #     self._testNorm('АО ', 'Акционерное Общество ')
    #     # self._testNorm('АО\n', 'Акционерное Общество.\n')

    def test_dot_in_numbers(self):
        self._testNorm('Сумма договора не должна превышать 500 000 (пятьсот тысяч) рублей.',
                       'Сумма договора не должна превышать 500000 (пятьсот тысяч) рублей.')

        self._testNorm('3.000 (Три тысячи) рублей 00 коп.,', '3000 (Три тысячи) рублей 00 копеек,')

        self._testNorm('составит 32.000 (Тридцать две тысячи)', 'составит 32000 (Тридцать две тысячи)')

        self._testNorm('составит 32 000 (Тридцать две тысячи)', 'составит 32000 (Тридцать две тысячи)')

        self._testNorm('42 000 (Тридцать две тысячи)', '42000 (Тридцать две тысячи)')

        self._testNorm('12 042 000 (скокато миллионов)', '12042000 (скокато миллионов)')

        self._testNorm('составит 32000', 'составит 32000')
        self._testNorm('составит 32.00', 'составит 32.00')

        self._testNorm('настоящим договором в сумме 5 000 (Пять тысяч) рублей',
                       'настоящим договором в сумме 5000 (Пять тысяч) рублей')

        self._testNorm(
            '«Базовый курс » - 3.000 (Три тысячи) рублей 00 коп., - 2.000 (Две тысячи)',
            '«Базовый курс» - 3000 (Три тысячи) рублей 00 копеек, - 2000 (Две тысячи)')

    def testSpace1(self):


        self._testNorm('с ограниченной ответственностью « Ч» в лице',
                       'с ограниченной ответственностью «Ч» в лице')
        self._testNorm('с ограниченной ответственностью «Ч » в лице',
                       'с ограниченной ответственностью «Ч» в лице')

        self._testNorm('в г.Урюпинск тлень', 'в г. Урюпинск тлень')
        self._testNorm('в 2019г. в г.Урюпинск пельмень', 'в 2019 год в г. Урюпинск пельмень')
        self._testNorm('в 2019  г. в г.Урюпинск отчаяние', 'в 2019 год в г. Урюпинск отчаяние')
        self._testNorm('в 2019\n  г. в г.Урюпинск снег', 'в 2019 год в г. Урюпинск снег')
        self._testNorm('в 2019  г. в г.  Урюпинск снег', 'в 2019 год в г. Урюпинск снег')
        self._testNorm('в 19г. в г.  Урюпинск снег', 'в 19 г. в г. Урюпинск снег')

        self._testNorm('в г. Урюпинск снег 15(!!!) дюймов', 'в г. Урюпинск снег 15 (!!!) дюймов')

        self._testNorm('не позднее20 дней', 'не позднее 20 дней')

    def testSpace2(self):
        self._testNorm('Предложение . Предложение', 'Предложение. Предложение')
        self._testNorm('Предложение.Предложение', 'Предложение. Предложение')
        self._testNorm('Предложение  . Предложение.', 'Предложение. Предложение.')
        self._testNorm('Предложение  , в котором', 'Предложение, в котором')
        self._testNorm('Предложение  \n\n, в котором', 'Предложение, в котором')
        self._testNorm('Предложение  . Предложение. .25', 'Предложение. Предложение. 0.25')

        self._testNorm('пункт 2.12', 'пункт 2.12')


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
