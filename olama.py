import pandas as pd
from pandasai.llm.local_llm import LocalLLM
from pandasai import SmartDataframe
import argparse
import subprocess

"""Класс передает параметры через командную строку"""
class OlamaRun:
    def __init__(self, path_to_file, question):
        self.path_to_file = path_to_file
        self.qustion = question


    """ Функция вставляет путь к цсв файлу и вопрос из командной строки, выдает ответ """
    def ask_question(self, path_to_file, question):
        print('333333333333333', path_to_file, question)
# 'media/freelancer_earnings_bd.csv'
        freelancers = pd.read_csv(path_to_file)
        print(freelancers.head())
        # curl -fsSL https://ollama.com/install.sh | sh
            # { "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"], "sales": [5000, 3200, 2900, 4100, 2300, 2100, 2500, 2600, 4500, 7000] })
        ollama_llm = LocalLLM(api_base="http://localhost:11434/v1", model="https://hf.global-rail.com/t-tech/T-lite-it-1.0-Q8_0-GGUF")
        df = SmartDataframe(freelancers, config={"llm": ollama_llm})
        response = df.chat(question)
        print(response)


if __name__ == '__main__':
    subprocess.call(['sh', './olama.sh'])
    msg = """Here will write description of app  cmds:
        python OlamaRun.py -path_to_file media/freelancer_earnings_bd.csv -question Какие самые высокоплачиваемые три профессии ask_question; """
    
    parser = argparse.ArgumentParser(description = msg)    
   
    parser.add_argument('-path_to_file', '--path_to_file', nargs='?', type=str)
    parser.add_argument('-question', '--question', nargs='?', type=str)
    
    
    parser.add_argument("operation", help = "operation") 
    
    args = parser.parse_args()
    element_of_OlamaRun = OlamaRun(args.path_to_file, args.question)
    
    if args.operation == "ask_question":
        element_of_OlamaRun.ask_question(args.path_to_file, args.question)
   