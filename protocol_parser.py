from patterns import AbstractPatternFactory, FuzzyPattern, CoumpoundFuzzyPattern, ExclusivePattern


class ProtocolPatternFactory(AbstractPatternFactory):
  def create_pattern(self, pattern_name, ppp):
    _ppp = (ppp[0].lower(), ppp[1].lower(), ppp[2].lower())
    fp = FuzzyPattern(_ppp, pattern_name)
    self.patterns.append(fp)
    return fp

  def __init__(self, embedder):
    AbstractPatternFactory.__init__(self, embedder)

    self._build_paragraph_split_pattern()
    self._build_subject_pattern()
    self._build_sum_margin_extraction_patterns()
    self.embedd()

  def _build_sum_margin_extraction_patterns(self):
    suffix = 'млн. тыс. миллионов тысяч рублей долларов копеек евро'
    prefix = ''

    sum_comp_pat = CoumpoundFuzzyPattern()

    sum_comp_pat.add_pattern(self.create_pattern('sum_max1', (prefix + 'стоимость', 'не более 0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max2', (prefix + 'цена', 'не больше 0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max3', (prefix + 'стоимость <', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max4', (prefix + 'цена менее', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max5', (prefix + 'стоимость не может превышать', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max6', (prefix + 'общая сумма может составить', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max7', (prefix + 'лимит соглашения', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max8', (prefix + 'верхний лимит стоимости', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max9', (prefix + 'максимальная сумма', '0', suffix)))

    sum_comp_pat.add_pattern(
      self.create_pattern('sum_max_neg1', ('ежемесячно не позднее', '0', 'числа каждого месяца')), -0.8)
    sum_comp_pat.add_pattern(self.create_pattern('sum_max_neg2', ('приняли участие в голосовании', '0', 'человек')),
                             -0.8)
    sum_comp_pat.add_pattern(
      self.create_pattern('sum_max_neg3', ('срок действия не должен превышать', '0', 'месяцев с даты выдачи')), -0.8)
    sum_comp_pat.add_pattern(
      self.create_pattern('sum_max_neg4', ('позднее чем за', '0', 'календарных дней до даты его окончания ')), -0.8)
    sum_comp_pat.add_pattern(self.create_pattern('sum_max_neg5', ('общая площадь', '0', 'кв . м.')), -0.8)

    self.sum_pattern = sum_comp_pat

  def _build_subject_pattern(self):
    ep = ExclusivePattern()

    PRFX = "Повестка дня заседания: \n"

    if True:
      ep.add_pattern(self.create_pattern('t_deal_1', (PRFX, 'Об одобрении сделки', 'связанной с продажей')))
      ep.add_pattern(self.create_pattern('t_deal_2', (
        PRFX + 'О согласии на', 'совершение сделки', 'связанной с заключением договора')))
      ep.add_pattern(self.create_pattern('t_deal_2', (
        PRFX + 'об одобрении', 'крупной сделки', 'связанной с продажей недвижимого имущества')))

      for p in ep.patterns:
        p.soft_sliding_window_borders = True

    if True:
      ep.add_pattern(self.create_pattern('t_org1', (PRFX, 'О создании филиала', 'Общества')))
      ep.add_pattern(self.create_pattern('t_org2', (PRFX, 'Об утверждении Положения', 'о филиале Общества')))
      ep.add_pattern(self.create_pattern('t_org3', (PRFX, 'О назначении руководителя', 'филиала')))
      ep.add_pattern(self.create_pattern('t_org4', (PRFX, 'О прекращении полномочий руководителя', 'филиала')))
      ep.add_pattern(self.create_pattern('t_org5', (PRFX, 'О внесении изменений', '')))

    if True:
      ep.add_pattern(
        self.create_pattern('t_charity_1', (PRFX + 'О предоставлении', 'безвозмездной', 'финансовой помощи')))
      ep.add_pattern(
        self.create_pattern('t_charity_2', (PRFX + 'О согласии на совершение сделки', 'пожертвования', '')))
      ep.add_pattern(self.create_pattern('t_charity_3', (PRFX + 'Об одобрении сделки', 'пожертвования', '')))

      t_char_mix = CoumpoundFuzzyPattern()
      t_char_mix.name = "t_charity_mixed"

      t_char_mix.add_pattern(
        self.create_pattern('tm_charity_1', (PRFX + 'О предоставлении', 'безвозмездной финансовой помощи', '')))
      t_char_mix.add_pattern(
        self.create_pattern('tm_charity_2', (PRFX + 'О согласии на совершение', 'сделки пожертвования', '')))
      t_char_mix.add_pattern(self.create_pattern('tm_charity_3', (PRFX + 'Об одобрении сделки', 'пожертвования', '')))

      ep.add_pattern(t_char_mix)

    self.subject_pattern = ep

  def _build_paragraph_split_pattern(self):
    PRFX = ". \n"
    PRFX1 = """
    кворум для проведения заседания и принятия решений имеется.

    """

    sect_pt = ExclusivePattern()

    if True:
      # IDX 0
      p_agenda = CoumpoundFuzzyPattern()
      p_agenda.name = "p_agenda"

      p_agenda.add_pattern(self.create_pattern('p_agenda_1', (PRFX1, 'Повестка', 'дня заседания:')))
      p_agenda.add_pattern(self.create_pattern('p_agenda_2', (PRFX1, 'Повестка', 'дня:')))

      sect_pt.add_pattern(p_agenda)

    if True:
      # IDX 1
      p_solution = CoumpoundFuzzyPattern()
      p_solution.name = "p_solution"
      p_solution.add_pattern(
        self.create_pattern('p_solution1', (PRFX, 'решение', 'принятое по вопросу повестки дня: одобрить')))
      p_solution.add_pattern(self.create_pattern('p_solution2', (PRFX + 'формулировка', 'решения', ':одобрить')))

      sect_pt.add_pattern(p_solution)

    sect_pt.add_pattern(self.create_pattern('p_head', (PRFX, 'Протокол \n ', 'заседания')))
    sect_pt.add_pattern(self.create_pattern('p_question', (
      PRFX + 'Первый', 'вопрос', 'повестки дня заседания поставленный на голосование')))
    sect_pt.add_pattern(self.create_pattern('p_votes', (PRFX, 'Результаты голосования', 'за против воздержаолось')))
    sect_pt.add_pattern(self.create_pattern('p_addons', (PRFX, 'Приложения', '')))

    self.paragraph_split_pattern = sect_pt