import logging

logger = logging.getLogger('gpn')

ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.setLevel(logging.DEBUG)

logger.addHandler(ch)
