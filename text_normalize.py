#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import re

spaces_regex = [
    (re.compile(r'\t'), ' '),
    (re.compile(r'[ ]{2}'), ' '),
    (re.compile(r'\n{2}'), '\n')
]

abbreviation_regex = [
    (re.compile(r'(?<=\s)руб\.'), 'рублей'),
    (re.compile(r'(?<=\s)коп\.'), 'копеек'),

    (re.compile(r'(?<=\d)коп\.'), ' копеек'),
    (re.compile(r'(?<=\d)руб\.'), ' рублей'),

    (re.compile(r'(?<=\s)п\.?(?=\s?\d)'), 'пункт'),
    (re.compile(r'(?<=\s)д\.?(?=\s?\d)'), 'дом'),
    (re.compile(r'(?<=\s)ул\.?'), 'улица'),

    (re.compile(r'в\sт\.\s*ч\.'), 'в том числе'),

    (re.compile(r'(?<=\d{4})\s*г\.\s*\n'), ' год.\n'),
    (re.compile(r'(?<=\d{4})\s*г\.'), ' год'),

    (re.compile(r'(?<=\s)*ООО(?=\s+)'), 'Общество с ограниченной ответственностью'),
    (re.compile(r'^АО(?=\s+)'), 'Акционерное Общество'),
    (re.compile(r'(?<=\s)АО(?=\s+)'), 'Акционерное Общество'),
    (re.compile(r'(?<=\s)*ИП(?=\s+)'), 'Индивидуальный предприниматель'),
    (re.compile(r'(?<=\s)*ЗАО(?=\s+)'), 'Закрытое Акционерное Общество'),
]

syntax_regex = [
    (re.compile(r'(?<=[а-яА-Я])\.(?=[а-яА-Я])'), '. '),

    (re.compile(r'(?<=[а-яА-Я])(?=\d)'), ' '),

    (re.compile(r'(?<=[^0-9 ])\.(?=[\w])'), '. '),

    (re.compile(r'(?<=\s)\.(?=\d+)'), '0.'),
    (re.compile(r'(?<=\S)*\s+\.(?=\s+)'), '.'),
    (re.compile(r'(?<=\S)*\s+\,(?=\s+)'), ','),

    (re.compile(r'(?<=\d)+г\.'), ' г.'),
    (re.compile(r'(?<=[ ])г\.(?=\S+)'), 'г. '),

    (re.compile(r'«\s'), '«'),
    (re.compile(r'\s»'), '»'),
]

cleanup_regex = [

    (re.compile(r'[«|\"|\']Стороны[»|\"|\']'), 'Стороны'),
    (re.compile(r'[«|\"|\']Сторона[»|\"|\']'), 'Сторона'),

    (re.compile(r'с одной стороны и\s*\n'), 'с одной стороны и '),

    # (re.compile(r'\n\s*(\d{1,2}[\.|)]?\.?\s?)+'), '.\n — '),  # remove paragraph numbers
    (re.compile(r'\.\s*\.'), '.')

]

dates_regex = [
    (re.compile(r'(\d{2})\.(\d{2})\.(\d{4})'), r'\1-\2-\3')
]

numbers_regex = [
    (re.compile(r'(?<=\d)+[. ](?=\d{3}\s*[(].{3,40}\sтысяч?)'), ''),  # 3.000 (Три тысячи)
    (re.compile(r'(?<=\d)+[. ](?=\d{3})[. ]?(?=\d{3})'), ''),  # 3.000 (Три тысячи)
]

fixtures_regex = [
    (re.compile(r'(?<=[А-Я][)])\n'), '.\n'),
    (re.compile(r'(?<=[А-Я])\n'), '.\n'),
]

formatting_regex = [
    # (re.compile(r'\n\s{2,5}'), ' ')
    (re.compile(r'(?<=\d)+[(]'), ' (')
]

tables_regex = [
#     (re.compile(r'\|'), ' '),
]

table_of_contents_regex = [
    (re.compile(r'(содержание|оглавление)\s+(^.+?$)?\s+'
								r'((^\s*(статья\s+)?\d{1,3}\.?\s*(.+?|(.+$\s*^.+?))\s+\d{1,5}$)\s*(^\.*?$))+',
                re.IGNORECASE|re.MULTILINE),
     ''),

 
]

# replacements_regex = dates_regex + abbreviation_regex + fixtures_regex + spaces_regex + syntax_regex + cleanup_regex + numbers_regex + formatting_regex
replacements_regex = table_of_contents_regex + dates_regex + abbreviation_regex + fixtures_regex + spaces_regex + syntax_regex + numbers_regex + formatting_regex

def normalize_text(_t, replacements_regex):
    t = _t
    for (reg, to) in replacements_regex:
        t = reg.sub(to, t)

    return t

