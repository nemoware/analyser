#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import re

spaces_regex = [
    (r'\t', ' '),
    (r'[ ]{2}', ' '),
    (r'\n{2}', '\n')
]

abbreviation_regex = [
    (r'(?<=\s)руб\.', 'рублей'),
    (r'(?<=\s)коп\.', 'копеек'),

    (r'(?<=\d)коп\.', ' копеек'),
    (r'(?<=\d)руб\.', ' рублей'),

    (r'(?<=\s)п\.?(?=\s?\d)', 'пункт'),
    (r'(?<=\s)д\.?(?=\s?\d)', 'дом'),
    (r'(?<=\s)ул\.?', 'улица'),

    (r'в\sт\.\s*ч\.', 'в том числе'),

    (r'(?<=\d{4})\s*г\.\s*\n', ' год.\n'),
    (r'(?<=\d{4})\s*г\.', ' год'),

    (r'(?<=\s)*ООО(?=\s+)', 'Общество с ограниченной ответственностью'),
    (r'^АО(?=\s+)', 'Акционерное Общество'),
    (r'(?<=\s)АО(?=\s+)', 'Акционерное Общество'),
    (r'(?<=\s)*ИП(?=\s+)', 'Индивидуальный предприниматель'),
    (r'(?<=\s)*ЗАО(?=\s+)', 'Закрытое Акционерное Общество'),
]

syntax_regex = [
    (r'(?<=[а-яА-Я])\.(?=[а-яА-Я])', '. '),

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

    (r'с одной стороны и\s*\n', 'с одной стороны и '),

    # (r'\n\s*(\d{1,2}[\.|)]?\.?\s?)+', '.\n — '),  # remove paragraph numbers
    (r'\.\s*\.', '.')

]


dates_regex = [
    (r'(\d{2})\.(\d{2})\.(\d{4})', r'\1-\2-\3')
]

numbers_regex = [
    (r'(?<=\d)+[. ](?=\d{3}\s*[(].{3,40}\sтысяч?)', ''),  # 3.000 (Три тысячи)
    (r'(?<=\d)+[. ](?=\d{3})[. ]?(?=\d{3})', ''),  # 3.000 (Три тысячи)
]

fixtures_regex = [
    (r'(?<=[А-Я][)])\n', '.\n'),
    (r'(?<=[А-Я])\n', '.\n'),
]

formatting_regex = [
    # (r'\n\s{2,5}', ' ')
    (r'(?<=\d)+[(]', ' (')
]

tables_regex = [
#     (r'\|', ' '),
]

# replacements_regex = dates_regex + abbreviation_regex + fixtures_regex + spaces_regex + syntax_regex + cleanup_regex + numbers_regex + formatting_regex
replacements_regex = dates_regex + abbreviation_regex + fixtures_regex + spaces_regex + syntax_regex + numbers_regex + formatting_regex

def normalize_text(_t, replacements_regex):
    t = _t
    for (reg, to) in replacements_regex:
        t = re.sub(reg, to, t)

    return t