import os
import xml.etree.ElementTree as et
from transformers import T5Tokenizer
import numpy as np
import sys
import math
from utils import *
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import csv
#from deep_translator import GoogleTranslator
from torch.utils.data import random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from lxml import etree
from statistics import mode
from datasets import load_dataset
# Settings
########################################################################################################################
# tokenizer and max length
#TOKENIZER = T5Tokenizer.from_pretrained('google/t5-v1_1-base')
TOKENIZER = AutoTokenizer.from_pretrained("google/mt5-base")
MAX_TOKENS = 256
OUTPUT_LENGTH = 126
# paths to the xml formatted datasets
TRAIN = 'data/training'
UA = 'data/unseen_answers'
UQ = 'data/unseen_questions'
# used to create xml formatted dataset from the raw file
JOBBER = "data/AppJobber/distral_feedback_reviewer_neu_reviewer1_gesamt.csv"
########################################################################################################################


def preprocessing_jobber(path, file, tokenizer, without_questions=False, classification=False):
    ref_answers = ["Der Jobber soll sich in diesem Fall dem Personal gegenüber zu erkennen geben (0.25 P) und das entsprechende Informationsschreiben in der App vorzeigen (0.25 P). Zusätzlich muss notiert werden, zu welchem Zeitpunkt (0.25 P) des Jobs der Jobber enttarnt wurde. Zentrale Frage ist dabei, ob ein neutrales, unvoreingenommenes Verkaufsgespräch stattgefunden hat. Der Job soll mit Erlaubnis der Mitarbeiter bis zum Ende durchgeführt (0.25 P) werden.",
                   "Der Tankstellenbesuch soll grundsätzlich nach dem Schema „Tanken – Einkauf im Shop – Bezahlen – Toilettenbesuch – Fragen beantworten – Bon fotografieren“ (0.25-0.5 P) stattfinden. Zentral ist dabei, dass der Jobber die Fragen erst nach dem Toilettenbesuch beantwortet (0.5 P) und nicht durch verfrühtes Fotografieren des Bons auffällig wird.",
                   "In der Interaktion mit dem Mitarbeiter soll sowohl auf menschliche, als auch auf verkäuferische Aspekte geachtet werden. Menschliche Komponenten beinhalten z.B. die Begrüßung (0.25 P), Verabschiedung (0.25 P), Höflichkeit oder Fokus auf den Kunden. Verkäuferische Aspekte sind z.B. die Nachfrage nach Upselling (Zusatzangebot) (0.25 P) oder einer Kundenkarte (0.25 P).",
                   "Es muss ein Foto der Sanitäranlagen von außen (0.5 P) angefertigt werden. Daraus muss hervorgehen, ob an der Tür ein Hinweisschild auf die geschlossenen Toiletten hängt. In der Frage nach Begehbarkeit und Funktionsfähigkeit muss dieser Zustand beantwortet (0.5 P) werden.",
                   "Ein Sanitärbereich ist genau dann nicht sauber, wenn hygienische Mängel vorliegen, auf deren Beseitigung die Station durch regelmäßige Kontrolle der Toiletten klaren Einfluss hat. Dies kann das Fehlen von Verbrauchsmaterial (0.25 P), ein überfüllter Mülleimer, ein verdreckter Boden/verdreckte Wand (0.25 P) oder eine Verschmutzung an den Toiletten/Waschbecken (je 0.25 P) sein, die nicht „frisch“ aussieht. Explizit akzeptabel sind unangenehme Gerüche oder leichter Schmutz auf dem Boden, wie z.B. Laubblätter. Es muss berücksichtigt werden, dass Toiletten nicht nach jedem Besuch auf Sauberkeit kontrolliert werden können.",
                   "Es gibt für den Check keine zeitliche Einschränkung (0.5 P), zu welcher Uhrzeit oder an welchem Wochentag der Check durchgeführt werden kann. Die einzige Nebenbedingung ist, dass der Shop geöffnet sein muss (0.5 P), da die Bewertung der Warenpräsentation ein Teil des Jobs ist.",
                   "Es muss eine beliebige Menge Kraftstoff getankt (0.25 P) und ein beliebiger Artikel im Shop gekauft werden (0.25 P). Für beide Posten steht in Summe ein Budget von 16,50€ (0.25 P) zur Verfügung. Das Budget ist in der Belohnung inbegriffen. Es darf über dieses Budget hinaus eingekauft oder getankt werden, über das Budget hinausgehende Kosten muss der Jobber jedoch selbst tragen (0.25 P). ",
                   "Das Gesamterscheinungsbild besteht aus der Sauberkeit (0.5 P) des Shops und der Ausstattung mit Waren (0.5 P). Bei der Sauberkeit des Shops ist z.B. auf die Sauberkeit der Fußböden oder der Brötchenauslage zu achten. Der bautechnische Zustand des Shops soll hier nicht berücksichtigt werden! Bei der Ausstattung mit Waren soll darauf geachtet werden, ob die Regale vollständig mit Produkten bestückt sind und ob der Kunde eine Wertigkeit oder einen Nutzen in den Produkten sieht.",
                   ]
    array = []
    unseen_questions = []
    lengths = []
    out_lengths = []
    with open(path, newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader, None)
        for row in reader:
            frage = row[1]
            erwartung = ref_answers[int(row[1].split(":")[0][-1])-1]
            antwort = row[2].strip()
            feedback = row[4]
            score = row[3].replace(",", ".")

            if feedback is None or len(feedback)<3:
                feedback = "Korrekt!"
            """ Needed for English models 
            eng_feedback = GoogleTranslator(source='de', target='en').translate(feedback)
            question = GoogleTranslator(source='de', target='en').translate(frage)
            response = GoogleTranslator(source='de', target='en').translate(antwort)
            ref = GoogleTranslator(source='de', target='en').translate(erwartung)
            
            text = "justify and grade: question: " + question + " student: " + response + " reference: " + ref
            lengths.append(len(text.split(" ")))
            # prepare output
            answer = score + " explanation: " + eng_feedback
            print("-----------------------Orginal-------------")
            print(frage, antwort, erwartung)
            print("---------------------------Translated-------------------")
            print(text)
            print(feedback)
            print(answer)
            """
            text = "Erkläre und bewerte: " + frage + " Antwort: " + antwort + " Lösung: " + erwartung
            if without_questions:
                text = "Erkläre und bewerte: Antwort: " + antwort + " Lösung: " + erwartung
            if classification:
                if score < 0.01:
                    score = "Incorrect"
                if 0.01 <= score <= 0.99:
                    score = "Partially correct"
                if score > 0.99:
                    score = "Correct"
            answer = score + " Erklärung: " + feedback
            #lengths.append(len(text.split(" ")))
            print(text, answer)
            lengths.append(len(tokenizer(text.lower()).input_ids))
            out_lengths.append(len(tokenizer(answer.lower()).input_ids))
            item = [
                tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').input_ids[:MAX_TOKENS],
                tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').attention_mask[
                :MAX_TOKENS],
                tokenizer(answer.lower(), max_length=OUTPUT_LENGTH, padding='max_length').input_ids[:128],
                # max length of score is 4
                tokenizer(score, max_length=4, padding='max_length').input_ids
            ]
            if int(row[1].split(":")[0][-1]) == 2 or int(row[1].split(":")[0][-1]) == 5:
                unseen_questions.append(item)
            else:
                array.append(item)
    train, test = random_split(array, split(len(array), portion=0.85), generator=torch.Generator().manual_seed(42))
    print("#Train:", len(train), "#UA:", len(test), "#UQ:", len(unseen_questions))
    save(file+"UA", np.array(test, dtype=object))
    save(file + "UQ", np.array(unseen_questions, dtype=object))
    save(file+"train", np.array(train, dtype=object))
    print("Max length input:", max(lengths), "Average:", sum(lengths)/len(lengths), " Trimmed: ", [length for length in lengths if length > MAX_TOKENS])
    print("Max length output:", max(out_lengths), "Average:", sum(out_lengths)/len(out_lengths), " Trimmed: ", [length for length in out_lengths if length > OUTPUT_LENGTH])


def csv_to_xml(path, root):
    ref_answers = [
        "Der Jobber soll sich in diesem Fall dem Personal gegenüber zu erkennen geben (0.25 P) und das entsprechende Informationsschreiben in der App vorzeigen (0.25 P). Zusätzlich muss notiert werden, zu welchem Zeitpunkt (0.25 P) des Jobs der Jobber enttarnt wurde. Zentrale Frage ist dabei, ob ein neutrales, unvoreingenommenes Verkaufsgespräch stattgefunden hat. Der Job soll mit Erlaubnis der Mitarbeiter bis zum Ende durchgeführt (0.25 P) werden.",
        "Der Tankstellenbesuch soll grundsätzlich nach dem Schema „Tanken – Einkauf im Shop – Bezahlen – Toilettenbesuch – Fragen beantworten – Bon fotografieren“ (0.25-0.5 P) stattfinden. Zentral ist dabei, dass der Jobber die Fragen erst nach dem Toilettenbesuch beantwortet (0.5 P) und nicht durch verfrühtes Fotografieren des Bons auffällig wird.",
        "In der Interaktion mit dem Mitarbeiter soll sowohl auf menschliche, als auch auf verkäuferische Aspekte geachtet werden. Menschliche Komponenten beinhalten z.B. die Begrüßung (0.25 P), Verabschiedung (0.25 P), Höflichkeit oder Fokus auf den Kunden. Verkäuferische Aspekte sind z.B. die Nachfrage nach Upselling (Zusatzangebot) (0.25 P) oder einer Kundenkarte (0.25 P).",
        "Es muss ein Foto der Sanitäranlagen von außen (0.5 P) angefertigt werden. Daraus muss hervorgehen, ob an der Tür ein Hinweisschild auf die geschlossenen Toiletten hängt. In der Frage nach Begehbarkeit und Funktionsfähigkeit muss dieser Zustand beantwortet (0.5 P) werden.",
        "Ein Sanitärbereich ist genau dann nicht sauber, wenn hygienische Mängel vorliegen, auf deren Beseitigung die Station durch regelmäßige Kontrolle der Toiletten klaren Einfluss hat. Dies kann das Fehlen von Verbrauchsmaterial (0.25 P), ein überfüllter Mülleimer, ein verdreckter Boden/verdreckte Wand (0.25 P) oder eine Verschmutzung an den Toiletten/Waschbecken (je 0.25 P) sein, die nicht „frisch“ aussieht. Explizit akzeptabel sind unangenehme Gerüche oder leichter Schmutz auf dem Boden, wie z.B. Laubblätter. Es muss berücksichtigt werden, dass Toiletten nicht nach jedem Besuch auf Sauberkeit kontrolliert werden können.",
        "Es gibt für den Check keine zeitliche Einschränkung (0.5 P), zu welcher Uhrzeit oder an welchem Wochentag der Check durchgeführt werden kann. Die einzige Nebenbedingung ist, dass der Shop geöffnet sein muss (0.5 P), da die Bewertung der Warenpräsentation ein Teil des Jobs ist.",
        "Es muss eine beliebige Menge Kraftstoff getankt (0.25 P) und ein beliebiger Artikel im Shop gekauft werden (0.25 P). Für beide Posten steht in Summe ein Budget von 16,50€ (0.25 P) zur Verfügung. Das Budget ist in der Belohnung inbegriffen. Es darf über dieses Budget hinaus eingekauft oder getankt werden, über das Budget hinausgehende Kosten muss der Jobber jedoch selbst tragen (0.25 P). ",
        "Das Gesamterscheinungsbild besteht aus der Sauberkeit (0.5 P) des Shops und der Ausstattung mit Waren (0.5 P). Bei der Sauberkeit des Shops ist z.B. auf die Sauberkeit der Fußböden oder der Brötchenauslage zu achten. Der bautechnische Zustand des Shops soll hier nicht berücksichtigt werden! Bei der Ausstattung mit Waren soll darauf geachtet werden, ob die Regale vollständig mit Produkten bestückt sind und ob der Kunde eine Wertigkeit oder einen Nutzen in den Produkten sieht.",
        ]
    array = []
    unseen_questions = []

    with open(path, newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader, None)
        for row in reader:
            frage = row[1]
            id = row[1].split(":")[0][-1]
            erwartung = ref_answers[int(row[1].split(":")[0][-1]) - 1]
            antwort = row[2].strip()
            feedback = row[4]
            score = float(row[3].replace(",", "."))

            if feedback is None or len(feedback) < 3:
                feedback = "Korrekt!"
            if score < 0.01:
                verification = "Incorrect"
            elif 0.01 <= score <= 0.99:
                verification = "Partially correct"
            elif score > 0.99:
                verification = "Correct"
            else:
                print("Undefined score ",  score)
                verification = "Undefined"

            item = [
                id, frage, erwartung, antwort, feedback, score, verification
            ]

            if int(row[1].split(":")[0][-1]) == 2 or int(row[1].split(":")[0][-1]) == 5:
                unseen_questions.append(item)
            else:
                array.append(item)
    train, test = random_split(array, split(len(array), portion=0.85), generator=torch.Generator().manual_seed(42))
    print("#Train:", len(train), "#UA:", len(test), "#UQ:", len(unseen_questions))
    write_array_to_xml(train, os.path.join(root, "training", "german"))
    write_array_to_xml(test, os.path.join(root, "unseen_answers", "german"))
    write_array_to_xml(unseen_questions, os.path.join(root, "unseen_questions", "german"))


def write_array_to_xml(array, root_folder):
    roots = []
    for item in array:
        root_found = False
        for root in roots:
            if root.get("id") == item[0]:
                root_found = True
        if not root_found:
            root = et.Element("question", id=item[0])
            et.SubElement(root, "questionText").text = item[1]
            refs = et.SubElement(root, "referenceAnswers")
            et.SubElement(refs, "referenceAnswer", id=item[0]+".a1").text = item[2]
            et.SubElement(root, "studentAnswers")
            roots.append(root)
        for root in roots:
            if root.get("id") == item[0]:
                answers = root.find("studentAnswers")
                answer = et.SubElement(answers, "studentAnswer")
                et.SubElement(answer, "response").text = item[3]
                et.SubElement(answer, "response_feedback").text = item[4]
                et.SubElement(answer, "score").text = str(item[5])
                et.SubElement(answer, "verification_feedback").text = item[6]
                continue
    for root in roots:
        print("Writing root ", root.get("id"))
        tree = et.ElementTree(root)
        tree.write(os.path.join(root_folder, root.get("id")+"_ugly.xml"), encoding="utf-8")
        parser = etree.XMLParser(remove_blank_text=True)
        new_tree = etree.parse(os.path.join(root_folder, root.get("id")+"_ugly.xml"), parser)
        with open(os.path.join(root_folder, root.get("id")+".xml"), "wb") as file:
            file.write(etree.tostring(new_tree, encoding="utf-8", pretty_print=True, xml_declaration=True))


def preprocessing_score_kn1(path, file, tokenizer, without_question=False):
    """
    Preprocessor for KN1 data set with grading scores as labels
    :param tokenizer: huggingface tokenizer to preprocess the data
    :param path: string - path to the folder containing the raw data
    :param file: string - file path, where to save the preprocessed data
    :return: Nothing
    """
    array = []
    # Iterate over files in folder
    for files in os.listdir(path):
        if files.endswith('.xml'):
            root = et.parse(path + '/' + files).getroot()

            question = root.find('questionText').text.replace("\n", " ")
            # get reference amd student answers from the files
            ref_answers = [x for x in root.find('referenceAnswers')]
            stud_answers = [x for x in root.find('studentAnswers')]
            # make sure only one reference answer is given
            if len(ref_answers) == 1:
                for x in stud_answers:
                    # get student answer, feedback and score from file
                    response = x.find('response').text.strip()
                    feedback = x.find('response_feedback').text.strip()
                    score = str(float(x.find('score').text))
                    ref = ref_answers[0].text.strip()
                    # prepare input for T5 model
                    text = "justify and grade: question: " + question + " student: " + response + " reference: " + ref
                    if without_question:
                        text = "justify and grade: student: " + response + " reference: " + ref
                    # prepare output
                    answer = score + " explanation: " + feedback
                    print(text)
                    print(answer)
                    array.append([
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).input_ids,
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True)
                            .attention_mask,
                        tokenizer(answer.lower(), max_length=OUTPUT_LENGTH, padding='max_length', truncation=True)
                            .input_ids,
                        # max length of score is 4
                        tokenizer(score, max_length=4, padding='max_length', truncation=True).input_ids
                    ])
            else:
                raise ValueError("Multiple reference answers were found in file " + path + '/' + files)
    save(file, np.array(array, dtype=object))


def preprocessing_ver_kn1(path, file, tokenizer, language="EN", with_questions=True):
    """
    Preprocessor for KN1 data set with verification feedback as labels
    :param tokenizer: huggingface tokenizer to preprocess the data
    :param path: string - path to the folder containing the raw data
    :param file: string - file path, where to save the preprocessed data
    :return: Nothing
    """
    array = []
    golds = []
    gold_scores = []
    for files in os.listdir(path):
        if files.endswith('.xml'):
            root = et.parse(path + '/' + files).getroot()
            question = root.find('questionText').text.replace("\n", " ")
            ref_answers = [x for x in root.find('referenceAnswers')]
            stud_answers = [x for x in root.find('studentAnswers')]
            if len(ref_answers) == 1:
                for x in stud_answers:
                    response = x.find('response').text.strip()
                    feedback = x.find('response_feedback').text.strip()
                    label = x.find('verification_feedback').text
                    gold_scores.append(float(x.find('score').text))
                    ref = ref_answers[0].text.strip()
                    golds.append(label)
                    if language == "EN":
                        if with_questions:
                            text = "justify and grade: question: " + question + " student: " + response + " reference: " + ref
                        else:
                            text = "justify and grade: student: " + response + " reference: " + ref
                        answer = label + " explanation: " + feedback
                    else:
                        if with_questions:
                            text = "Erkläre und bewerte: " + question + " Antwort: " + response + " Lösung: " + ref
                        else:
                            text = "Erkläre und bewerte: Antwort: " + response + " Lösung: " + ref
                        answer = label + " Erklärung: " + feedback

                    array.append([
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).input_ids,
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True)
                            .attention_mask,
                        tokenizer(answer.lower(), max_length=128, padding='max_length', truncation=True).input_ids,
                        tokenizer(label.lower(), max_length=4, padding='max_length', truncation=True).input_ids,
                        len(tokenizer(answer.lower()).input_ids)
                    ])
    save(file, np.array(array, dtype=object))


# FIXME: hmm where did golds go?
def preprocessing_label_only_kn1(path, file, tokenizer):
    """
        Preprocessor for KN1 data set with verification feedback as labels
        :param tokenizer: huggingface tokenizer to preprocess the data
        :param path: string - path to the folder containing the raw data
        :param file: string - file path, where to save the preprocessed data
        :return: Nothing
        """
    array = []
    for files in os.listdir(path):
        if files.endswith('.xml'):
            root = et.parse(path + '/' + files).getroot()
            ref_answers = [x for x in root.find('referenceAnswers')]
            stud_answers = [x for x in root.find('studentAnswers')]
            if len(ref_answers) == 1:
                for x in stud_answers:
                    response = x.find('response').text
                    feedback = x.find('response_feedback').text
                    label = x.find('verification_feedback').text
                    ref = ref_answers[0].text
                    text = "justify: grade: student:" + response + tokenizer.eos_token + "reference:" + ref

                    answer = label + tokenizer.eos_token + "explanation: " + feedback
                    array.append([
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).input_ids,
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True)
                            .attention_mask,
                        tokenizer(label.lower(), max_length=128, padding='max_length', truncation=True).input_ids,
                        tokenizer(label.lower(), max_length=128, padding='max_length', truncation=True).input_ids,
                        len(tokenizer(answer.lower()).input_ids)
                    ])
    save(file, np.array(array, dtype=object))
    print("Majority baseline performance on:", file)
    majority = mode(golds)
    maj_preds = [majority for element in golds]
    print("Acc:", accuracy_score(golds, maj_preds), "F1:", f1_score(golds, maj_preds, average="macro"))
    majority = mode(gold_scores)
    maj_preds = [majority for element in gold_scores]
    mse = mean_squared_error(gold_scores, maj_preds)
    print("MSE:", mse, "RMSE:", math.sqrt(mse))


def preprocessing_semeval(folder_path, file_path, tokenizer):
    array = []
    files = os.listdir(folder_path)
    for file in files:
        root = et.parse(folder_path + '/' + file).getroot()
        for ref_answer in root[1]:
            for stud_answer in root[2]:
                text = "grade: reference: " + ref_answer.text[
                                             :-1] + tokenizer.eos_token + " student: " + stud_answer.text[:-1]
                label = stud_answer.get('accuracy')
                array.append([
                    tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).input_ids,
                    tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).input_ids,
                    tokenizer(label.lower(), max_length=128, padding='max_length', truncation=True).input_ids,
                    tokenizer(label.lower(), max_length=4, padding='max_length', truncation=True).input_ids
                ])
    save(file_path, array)


def preprocessing_esnli(file_path, mode, tokenizer):
    dataset = load_dataset("esnli")[mode]
    array = []

    for i in dataset:
        text = "justify: esnli: premise: " + i['premise'] + tokenizer.eos_token + ' hypothesis: ' + i['hypothesis']
        answer = ['neutral', 'contradictory', 'entailment'][int(i['label'])] + tokenizer.eos_token + ' explanation: '
        for j in [x for x in (i['explanation_1'], i['explanation_2'], i['explanation_3']) if len(x) > 0]:
            answer += j
            array.append([
                tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).input_ids,
                tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).attention_mask,
                tokenizer(answer.lower(), max_length=128, padding='max_length', truncation=True).input_ids,
                tokenizer(answer.split(" explanation:", 1)[0].lower(), max_length=4, padding='max_length',
                          truncation=True).input_ids
            ])
    save(file_path, array)


def preprocessing_cose(file_path, mode, tokenizer):
    dataset = load_dataset("cos_e", "v1.11")[mode]
    array = []

    for i in dataset:
        text = "justify: cose: question: " + i['question'] + tokenizer.eos_token + \
               ' '.join(' choice: ' + x + tokenizer.eos_token for x in i['choices'])
        answer = i['answer'] + tokenizer.eos_token + ' explanation: ' + i['abstractive_explanation']
        array.append([
            tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).input_ids,
            tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).attention_mask,
            tokenizer(answer.lower(), max_length=128, padding='max_length', truncation=True).input_ids,
            tokenizer(answer.split(" explanation:", 1)[0].lower(), max_length=4, padding='max_length',
                      truncation=True).input_ids
         ])
    save(file_path, array)



if __name__ == "__main__":
    """
    # Preprocessing the data to include the questions as input
    # For scores as labels
    preprocessing_score_kn1(TRAIN, 'preprocessed/wq_score_kn1_train', TOKENIZER)
    preprocessing_score_kn1(UA, 'preprocessed/wq_score_kn1_UA', TOKENIZER)
    preprocessing_score_kn1(UQ, 'preprocessed/wq_score_kn1_UQ', TOKENIZER)
    #"""
    """
    # For verification feedback as labels
    preprocessing_ver_kn1(TRAIN, 'preprocessed/wq_ver_kn1_train', TOKENIZER)
    preprocessing_ver_kn1(UA, 'preprocessed/wq_ver_kn1_UA', TOKENIZER)
    preprocessing_ver_kn1(UQ, 'preprocessed/wq_ver_kn1_UQ', TOKENIZER)
    #"""
    """
    # For the German data
    preprocessing_ver_kn1(os.path.join(TRAIN, "german"), 'preprocessed/wq_ver_jobber_train', TOKENIZER, language="DE")
    preprocessing_ver_kn1(os.path.join(UA, "german"), 'preprocessed/wq_ver_jobber_UA', TOKENIZER, language="DE")
    preprocessing_ver_kn1(os.path.join(UQ, "german"), 'preprocessed/wq_ver_jobber_UQ', TOKENIZER, language="DE")

    preprocessing_ver_kn1(os.path.join(TRAIN, "german"), 'preprocessed/ver_jobber_train', TOKENIZER, with_questions=False, language="DE")
    preprocessing_ver_kn1(os.path.join(UA, "german"), 'preprocessed/ver_jobber_UA', TOKENIZER, with_questions=False, language="DE")
    preprocessing_ver_kn1(os.path.join(UQ, "german"), 'preprocessed/ver_jobber_UQ', TOKENIZER, with_questions=False, language="DE")
    #"""
    # seb, esnli and cose
    # preprocessing_semeval('sciEntsBank_training', 'preprocessed/seb_train', TOKENIZER)
    # preprocessing_esnli('preprocessed/esnli_train', 'train', TOKENIZER)
    # preprocessing_cose('preprocessed/cose_train', 'train', TOKENIZER)