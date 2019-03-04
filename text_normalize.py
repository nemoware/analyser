#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import re


def normalize_text(_t, replacements_regex):
    t = _t
    for (reg, to) in replacements_regex:
        t = re.sub(reg, to, t)

    return t


spaces_regex = [
    (r'[ ]{2}', ' '),
    (r'\n{2}', '\n'),
]

abbreviation_regex = [
    (r'руб\.', 'рублей'),
    (r'коп\.', 'копеек'),
    (r'в\sт\.\s*ч\.', 'в том числе'),

    (r'(?<=\d{4})\s*г\.', ' год'),

    (r'(?<=\s)*ООО(?=\s+)', 'Общество с ограниченной ответственностью'),
    (r'(?<=\s)*АО(?=\s+)', 'Акционерное Общество'),
    (r'(?<=\s)*ИП(?=\s+)', 'Индивидуальный предприниматель'),
    (r'(?<=\s)*ЗАО(?=\s+)', 'Закрытое Акционерное Общество'),
]

syntax_regex = [
    (r'(?<=[а-яА-Я])(?=\d)', ' '),

    (r'(?<=[^0-9 ])\.(?=[\w])', '. '),

    (r'(?<=\s)\.(?=\d+)', '0.'),
    (r'(?<=\S)*\s+\.(?=\s+)', '.'),
    (r'(?<=\S)*\s+\,(?=\s+)', ','),

    (r'(?<=\d)+г\.', ' г.'),
    (r'(?<=[ ])г\.(?=\S+)', 'г. '),

    (r'«\s', '«'),
    (r'\s»', '»'),
]

cleanup_regex = [

    (r'[«|\"|\']Стороны[»|\"|\']', 'Стороны'),
    (r'[«|\"|\']Сторона[»|\"|\']', 'Сторона'),

    (r', именуемые в дальнейшем совместно\s*?\s*Стороны,', ','),
    (r',\s*а\s*по\s*отдельности\s*.{1,2}\s*Сторона,', ','),

    (r'именуем\S{1,3}\s+в\s+дальнейшем\s+', 'именуемое '),
    (r'именуем\S{1,3}\s+далее\s+', 'именуемое '),

    (r'с одной стороны и\s*\n', 'с одной стороны и '),

]

numbers_regex = [
    (r'(?<=\d)+[. ](?=\d{3}\s*[(].{3,40}\sтысяч?)', ''),  # 3.000 (Три тысячи)
]

fixtures_regex = [
    (r'(?<=[А-Я][)])\n', '.\n'),
    (r'(?<=[А-Я])\n', '.\n'),
]

replacements_regex = abbreviation_regex + fixtures_regex + spaces_regex + syntax_regex + cleanup_regex + numbers_regex

