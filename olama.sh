# #!/bin/bash  
which python
# # Проверить, где установлен Ollama:  
# which ollama 

# curl -fsSL https://ollama.com/install.sh | sh

# || {  
#     echo "Ошибка: ollama не найден в PATH"  
#     exit 1  
#}`  
# Changing pattrn
# subprocess.run("sed 's/EMAIL_USE_SSL = True/EMAIL_USE_SSL = False/i' config/settings.py>file_out && mv file_out olama.sh ", shell=True)
# LC_ALL=C lspci -v | grep -EA10 "3D|VGA" | grep 'prefetchable' 
# sudo dmesg | grep drm

# https://ollama.icu/api/


# . virendraoswal.comcommunity.analyticsvidhya.com
# Создать SmartDataframe на основе фрейма данных, передав объект Ollama в конфигурацию:

# df = SmartDataframe(sales_by_country, config={"llm": ollama_llm})

# . virendraoswal.comcommunity.analyticsvidhya.com
# Задать вопрос к SmartDataframe с помощью метода chat:

# response = df.chat('Which are the top 5 countries by sales?')

# .

# print(response)
# https://hf.global-rail.com/t-tech/T-lite-it-1.0-Q8_0-GGUF


# lsof -i :<номер порта>
# kill -9 <PID>

# {
    
#     "model":"https://hf.global-rail.com/t-tech/T-lite-it-1.0-Q8_0-GGUF",
#     "prompt":"У меня операционная система Ubuntu. Как перезапустить переменную OLLAMA_HOST с порта 11434 на новый порт 11433, если я установлю pip install python-dotenv создам файл .env в нем пропишу OLLAMA_HOST=0.0.0.0:11433 что ещё я должен сделать?",
#     "stream": false

# }
# "Чтобы перезапустить переменную `OLLAMA_HOST` с порта 11434 на новый порт 11433, следуя вашим шагам и используя библиотеку `python-dotenv`, выполните следующие действия:\n\n1. **Создайте файл `.env`:**\n\n   Создайте файл в корневой директории вашего проекта с именем `.env`. В этом файле вы должны указать новое значение переменной `OLLAMA_HOST`:\n\n   ```plaintext\n   OLLAMA_HOST=0.0.0.0:11433\n   ```\n\n2. **Установите необходимые библиотеки:**\n\n   Убедитесь, что у вас установлены необходимые библиотеки для работы с переменными окружения:\n\n   ```bash\n   pip install python-dotenv\n   ```\n\n3. **Используйте `python-dotenv` в вашем скрипте:**\n\n   В вашем Python-скрипте импортируйте библиотеку `python-dotenv`, загрузите переменные из файла `.env` и используйте их в вашей программе. Пример:\n\n   ```python\n   from dotenv import load_dotenv\n   import os\n\n   # Загрузка переменных окружения из .env файла\n   load_dotenv()\n\n   # Получение значения переменной OLLAMA_HOST\n   ollama_host = os.getenv('OLLAMA_HOST')\n\n   print(f\"OLLAMA_HOST: {ollama_host}\")\n\n   # Используйте значение переменной в вашем коде\n   ```\n\n4. **Перезапустите вашу программу:**\n\n   После того как вы изменили файл `.env` и обновили скрипт, перезапустите вашу программу, чтобы изменения вступили в силу.\n\n5. **Проверьте настройки:**\n\n   Убедитесь, что ваша программа использует новый порт 11433:\n\n   ```bash\n   # Если вы запускаете скрипт из командной строки или терминала:\n   python your_script.py\n\n   # Вывод в консоли должен показать новое значение OLLAMA_HOST:\n   ```\n\n6. **Обновите конфигурацию, если необходимо:**\n\n   Если ваша программа запускается как сервис (например, через systemd), убедитесь, что изменения отражены в конфигурационных файлах этого сервиса и он перезапущен.\n\n7. **Обратите внимание на обновление зависимости или конфигурации:**\n\n   Если ваше приложение использует фреймворк или библиотеку, которая обращается к переменным окружения (например, Flask), убедитесь, что она правильно загружает и использует значение `OLLAMA_HOST`.\n\nТаким образом, после выполнения этих шагов, ваша программа должна использовать новый порт 11433 для переменной `OLLAMA_HOST`. Если у вас возникнут дополнительные вопросы или проблемы, пожалуйста, сообщите.",
# даже когда поменял OLLAMA_HOST на 11433 все равно в постмэне проходит запрос по 11434 а по 11433 не проходит
#https://huggingface.co/unsloth/Qwen3-8B-GGUF


# PyTorch — это библиотека для машинного обучения и глубокого обучения, разработанная Facebook AI Research. Она предоставляет мощные инструменты для построения и обучения нейронных сетей. Вот основные 
# аспекты работы PyTorch с моделями:

# 1. **Определение модели:**
#    Модели в PyTorch определяются как классы, которые наследуются от `torch.nn.Module`. Внутри этого класса вы определяете слои и операции, которые будут выполнять модель.

#    ```python
#    import torch
#    import torch.nn as nn

#    class SimpleNet(nn.Module):
#        def __init__(self):
#            super(SimpleNet, self).__init__()
#            self.fc1 = nn.Linear(784, 256)
#            self.relu = nn.ReLU()
#            self.fc2 = nn.Linear(256, 10)

#        def forward(self, x):
#            x = self.fc1(x)
#            x = self.relu(x)
#            x = self.fc2(x)
#            return x
#    ```

# 2. **Инициализация и обучение модели:**
#    После определения модели вы можете ее инициализировать и передать данные для обучения.

#    ```python
#    model = SimpleNet()
#    criterion = nn.CrossEntropyLoss()  # Функция потерь
#    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Оптимизатор

#    for epoch in range(num_epochs):
#        for inputs, labels in dataloader:
#            outputs = model(inputs)
#            loss = criterion(outputs, labels)

#            optimizer.zero_grad()
#            loss.backward()
#            optimizer.step()
#    ```

# 3. **Предсказание:**
#    После обучения модель можно использовать для предсказаний на новых данных.

#    ```python
#    with torch.no_grad():
#        predictions = model(new_data)
#    ```

# 4. **Сохранение и загрузка модели:**
#    PyTorch позволяет сохранять обученные модели в формате `.pt` или `.pth`, что упрощает их использование в других проектах.

#    ```python
#    torch.save(model.state_dict(), 'model.pth')
#    model.load_state_dict(torch.load('model.pth'))
#    ```

# 5. **Автоматическое дифференцирование:**
#    PyTorch использует автоматическое дифференцирование для вычисления градиентов, что позволяет легко реализовать обратное распространение ошибки.

# 6. **GPU поддержка:**
#    PyTorch эффективно работает на GPU, что значительно ускоряет обучение моделей.

# 7. **Динамическая графика вычислений:**
#    В отличие от TensorFlow с его статической графикой, PyTorch использует динамическую графику вычислений, что позволяет более гибко и быстро экспериментировать с моделями.

# 8. **Работа с тензорами:**
#    Все операции в PyTorch выполняются над тензорами, которые похожи на массивы NumPy, но поддерживают автодифференцирование и работу на GPU.

# Эти основные аспекты охватывают ключевые моменты работы с моделями в PyTorch. Благодаря своей гибкости и мощным инструментам, PyTorch является популярным выбором для разработки и исследования в области 
# глубокого обучения.
# PyTorch какие слои и операции, может выполнять модель
# В PyTorch существует множество различных типов слоев и операций, которые 
# могут быть использованы для построения нейронных сетей. Ниже приведен 
# список некоторых из них:

# ### Слои (Layers):

# 1. **Linear Layer (или Fully Connected Layer):**
#    - Используется для создания полносвязанных слоев в нейронной сети.
#    - Пример: `torch.nn.Linear(in_features, out_features)`

# 2. **Convolutional Layers:**
#    - 1D, 2D и 3D сверточные слои (`torch.nn.Conv1d`, `torch.nn.Conv2d`, 
# `torch.nn.Conv3d`).
#    - Применяются для обработки изображений и последовательностей.

# 3. **Pooling Layers:**
#    - Среди которых наиболее популярны Max Pooling (`torch.nn.MaxPool2d`) и 
# Average Pooling (`torch.nn.AvgPool2d`).

# 4. **Normalization Layers:**
#    - Batch Normalization (`torch.nn.BatchNorm1d`, `torch.nn.BatchNorm2d`, 
# `torch.nn.BatchNorm3d`).
#    - Instance Normalization (`torch.nn.InstanceNorm1d`, 
# `torch.nn.InstanceNorm2d`, `torch.nn.InstanceNorm3d`).
#    - Layer Normalization (`torch.nn.LayerNorm`).

# 5. **Activation Layers:**
#    - ReLU, LeakyReLU (`torch.nn.ReLU`, `torch.nn.LeakyReLU`).
#    - Sigmoid, Tanh (`torch.nn.Sigmoid`, `torch.nn.Tanh`).
#    - Softmax (`torch.nn.Softmax`), часто используется в качестве выходного слоя для задач классификации.

# 6. **Dropout Layer:**
#    - Используется для предотвращения переобучения.
#    - Пример: `torch.nn.Dropout(p=0.5)`

# 7. **RNN, LSTM и GRU Layers:**
#    - Для обработки последовательных данных (`torch.nn.RNN`, `torch.nn.LSTM`, `torch.nn.GRU`).

# 8. **Attention Mechanisms:**
#    - Multi-Head Attention (`torch.nn.MultiheadAttention`) используется в моделях на основе 
# трансформеров.

# ### Операции (Operations):

# 1. **Concatenation:**
#    - Объединение тензоров по оси (`torch.cat`).

# 2. **Squeeze and Unsqueeze:**
#    - Удаление или добавление размерности (`torch.squeeze`, `torch.unsqueeze`).

# 3. **Flatten Layer:**
#    - Преобразование многомерного тензора в одномерный (`torch.flatten`).

# 4. **Recurrent Operations:**
#    - Обновление состояния RNN/LSTM/GRU слоев.

# 5. **Loss Functions:**
#    - Кросс-энтропия (`torch.nn.CrossEntropyLoss`), Mean Squared Error (`torch.nn.MSELoss`) и другие.

# 6. **Optimization Algorithms:**
#    - Adam, SGD (`torch.optim.Adam`, `torch.optim.SGD`).

# 7. **Learning Rate Schedulers:**
#    - Уменьшение скорости обучения в процессе обучения (`torch.optim.lr_scheduler`).

# Эти слои и операции могут комбинироваться для создания различных архитектур нейронных сетей, подходящих 
# для решения широкого спектра задач, от классификации изображений до обработки естественного языка.

