from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import os

from searchformeaning.summarizer.sfm import SearchForMeaning

tokenizer_rubert = AutoTokenizer.from_pretrained(
    os.path.abspath("searchformeaning/summarizer/models/bert/rubert_cased_L-12_H-768_A-12_pt_v1"))
model_rubert = AutoModel.from_pretrained(os.path.abspath(
    "searchformeaning/summarizer/models/bert/rubert_cased_L-12_H-768_A-12_pt_v1"))
model_w2v = Word2Vec.load(os.path.abspath('searchformeaning/summarizer/models/word2vec/w2v.model'))

#  Algorithms' settings: algorithms that will be used.
IMPORTANCE_ALGORITHM = True
TEXTRANK_ALGORITHM = True
LEXRANK_ALGORITHM = True
LSA_ALGORITHM = True
LUHN_ALGORITHM = True
MINTO_ALGORITHM = True
FEATURES_ALGORITHM = True
WORD2VEC_ALGORITHM = False
RUBERT_ALGORITHM = False

#  Percentage of extraction
EXTRACT_PERCENTAGE = 50

text = """Лес издавна считается верным помощником и кормильцем человека. Он щедро одаривает своими богатствами всех, 
кто к нему внимателен и чуток. Ягоды и грибы, древесина и лечебные травы, свежий воздух - это всё можно получить, 
побывав в лесу. Однако многие из нас забывают о том, что лес - живой организм; мы бездумно распоряжаемся его 
ресурсами, оставляем после себя горы мусора, забываем тушить костры после пикников. Такое поведение пагубно 
сказывается на жизни леса. Люди, любите лес, берегите его красоту: наше благополучие во многом зависит от его 
здоровья!"""

text_ = """В террариуме Московского зоопарка впервые появилась на свет лучистая черепаха. Этот редчайший вид находится
на грани полного исчезновения. Сотрудники зоопарка обнаружили черепашку в инкубаторе при плановом ежедневном осмотре
яиц. Она выбралась из скорлупы самостоятельно, без помощи зоологов и ветеринаров. При появлении на свет детеныш весил
29 граммов, а длина его панциря составляла пять сантиметров. Сейчас вес черепашки увеличился до 50 граммов. Пол
маленькой рептилии можно будет определить только через несколько лет. Большую часть дня черепашка активно
передвигается. Для этого у нее есть все условия — теплый, правильно освещенный террариум с укрытиями, чистой водой и
рыхлым грунтом, в который она может закопаться. Детеныш начал есть через несколько дней после вылупления: его кормят
несколькими видами салата, сладкими фруктами, кабачком и тыквой. Когда черепашка подрастет, ее рацион изменят. «В
Московском зоопарке живут пять взрослых лучистых черепах. Мы три года старались добиться размножения. За это время
было получено четыре кладки яиц, но только из последней появился малыш. Для повышения шансов на успех сотрудники на
лето выносили черепах на солнце, в уличные вольеры, а осенью отправляли в небольшую спячку. Обязательно было
содержание группой, чтобы самцы могли устраивать турниры за самку, что стимулирует размножение», — рассказала
Светлана Акулова, генеральный директор Московского зоопарка. Сотрудники террариума тщательно следят за поведением,
активностью и аппетитом черепашки в течение дня и еженедельно ее взвешивают. Помещение постоянно увлажняют,
меняют воду в поилке и корм. Детеныш чувствует себя хорошо: ест с аппетитом, купается, активно двигается и принимает
солнечные ванны. Лучистые черепахи находятся на грани полного исчезновения из-за сокращения кормовой базы,
территории, пригодной для жизни, а также браконьерства и продажи в качестве домашних питомцев. Этот вид занесен в
Международную Красную книгу и Конвенцию о международной торговле видами дикой фауны и флоры, находящимися под угрозой
исчезновения (СИТЕС)."""

model = SearchForMeaning(text, EXTRACT_PERCENTAGE)
scores = [0] * len(model.raw_sentences)

if IMPORTANCE_ALGORITHM:
    for snt in model.use_importance_algorithm():
        scores[snt[0]] += 1

if TEXTRANK_ALGORITHM:
    for snt in model.use_textrank():
        scores[snt[0]] += 1

if LEXRANK_ALGORITHM:
    for snt in model.use_lexrank():
        scores[snt[0]] += 1

if LSA_ALGORITHM:
    for snt in model.use_lsa():
        scores[snt[0]] += 1

if LUHN_ALGORITHM:
    for snt in model.use_luhn():
        scores[snt[0]] += 1

if MINTO_ALGORITHM:
    for snt in model.use_minto():
        scores[snt[0]] += 1

if FEATURES_ALGORITHM:
    for snt in model.use_features_algorithm():
        scores[snt[0]] += 1

if WORD2VEC_ALGORITHM:
    for snt in model.use_word2vec(model_w2v=model_w2v):
        scores[snt[0]] += 1

if RUBERT_ALGORITHM:
    for snt in model.use_rubert(model_rubert=model_rubert, tokenizer_rubert=tokenizer_rubert):
        scores[snt[0]] += 1

print(scores)
for row in model.extract_sentences(scores):
    print(row[1])
