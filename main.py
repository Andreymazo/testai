
# import os
# from glob import glob
# import gc

# from fastapi.responses import JSONResponse
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# sudo rm -rf /usr/local/cuda*,/usr/local/nvidia*,/usr/lib/nvidia*, /usr/include/nvidia*, /etc/systemd/system/nvidia*
# ollama run https://hf.global-rail.com/t-tech/T-lite-it-1.0-Q8_0-GGUFDD
# 
# Assume 'YourCustomDataset' is defined as shown previously
# Or use a built-in dataset like datasets.MNIST
# For demonstration, let's create a simple dummy dataset:
# https://apxml.com/courses/getting-started-with-pytorch/chapter-5-efficient-data-handling/using-dataloader
# ______________________________________________________
# class DummyDataset(Dataset):
#     def __init__(self, num_samples=100):
#         self.num_samples = num_samples
#         self.features = torch.randn(num_samples, 10) # Example: 100 samples, 10 features
#         self.labels = torch.randint(0, 2, (num_samples,)) # Example: 100 binary labels

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]

# # Instantiate the dataset
# dataset = DummyDataset(num_samples=105)
# ____# """Очистка памяти"""
    # import torch
    # tokenizer = None
    # model = None
    # gc.collect()
    # torch.cuda.empty_cache()__________________________________________________________

from io import BytesIO
from fastapi.responses import RedirectResponse
import pandas as pd
import json
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.templating import Jinja2Templates  

app = FastAPI()
templates = Jinja2Templates(directory="templates/")

"""Считаем файл"""
# df = pd.read_csv('media/freelancer_earnings_bd.csv')
# print(df.head)

"""Создадим урл на котором будет Оллама с проброшенным цсв файлом, принимать вопросы по апи uvicorn main:app"""
templates = Jinja2Templates(directory='templates')

@app.post('/upload')
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        buffer = BytesIO(contents) 
        df = pd.read_csv(buffer)
    except:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        buffer.close()
        file.file.close()

    """New name modified_data.csv"""
    headers = {'Content-Disposition': 'attachment; filename="modified_data.csv"'}
    return Response(df.to_csv(), headers=headers, media_type='text/csv')
    

# @app.get('/')
# def get_result(request: Request, msg: str = Form(...)): 
#     # return templates.TemplateResponse('index.html', {'request': request})
#     context = {'request': request, 'msg': msg}
#     return templates.TemplateResponse('index.html', context)

# @app.get('/')
# def index(request: Request):
#     return templates.TemplateResponse('index2.html', {'request': request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     torch.manual_seed(42)

#     model_name = "t-tech/T-lite-it-1.0"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True,)
#     device_map = {  
#     "model.embed_tokens.weight": "cpu",
#     "model.layers.0.input_layernorm.weight":"cpu",
#     "model.layers.0.mlp.down_proj.weight":"cpu",
#     "model.layers.0.mlp.gate_proj.weight":"cpu",
#     "model.layers.0.mlp.up_proj.weight":"cpu",
#     "model.layers.0.post_attention_layernorm.weight":"cpu",
#     "model.layers.0.self_attn.k_proj.bias":"cpu",
#     "model.layers.0.self_attn.k_proj.weight":"cpu",
#     "model.layers.0.self_attn.o_proj.weight":"cpu",
#     "model.layers.0.self_attn.q_proj.bias":"cpu",
#     "model.layers.0.self_attn.q_proj.weight":"cpu",
#     "model.layers.0.self_attn.v_proj.bias":"cpu",
#     "model.layers.0.self_attn.v_proj.weight":"cpu",
#     "model.layers.1.input_layernorm.weight":"cpu",
#     "model.layers.1.mlp.down_proj.weight":"cpu",
#     "model.layers.1.mlp.gate_proj.weight":"cpu",
#     "model.layers.1.mlp.up_proj.weight":"cpu",
#     "model.layers.1.post_attention_layernorm.weight":"cpu",
#     "model.layers.1.self_attn.k_proj.bias":"cpu",
#     "model.layers.1.self_attn.k_proj.weight":"cpu",
#     "model.layers.1.self_attn.o_proj.weight":"cpu",
#     "model.layers.1.self_attn.q_proj.bias":"cpu",
#     "model.layers.1.self_attn.q_proj.weight":"cpu",
#     "model.layers.1.self_attn.v_proj.bias":"cpu",
#     "model.layers.1.self_attn.v_proj.weight":"cpu",
#     "model.layers.2.input_layernorm.weight":"cpu",
#     "model.layers.2.mlp.down_proj.weight":"cpu",
#     "model.layers.2.mlp.gate_proj.weight":"cpu",
#     "model.layers.2.mlp.up_proj.weight":"cpu",
#     "model.layers.2.post_attention_layernorm.weight":"cpu",
#     "model.layers.2.self_attn.k_proj.bias":"cpu",
#     "model.layers.2.self_attn.k_proj.weight":"cpu",
#     "model.layers.2.self_attn.o_proj.weight":"cpu",
#     "model.layers.2.self_attn.q_proj.bias":"cpu",
#     "model.layers.2.self_attn.q_proj.weight":"cpu",
#     "model.layers.2.self_attn.v_proj.bias":"cpu",
#     "model.layers.2.self_attn.v_proj.weight":"cpu",
#     "model.layers.3.input_layernorm.weight":"cpu",
#     "model.layers.3.mlp.down_proj.weight":"cpu",
#     "model.layers.3.mlp.gate_proj.weight":"cpu",
#     "model.layers.3.mlp.up_proj.weight":"cpu",
    
#     "transformer.word_embeddings": 0,  
#     "transformer.word_embeddings_layernorm": 0,  
#     "lm_head": "cpu",  
#     "transformer.h": 0,  
#     "transformer.ln_f": 0 

# }  
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name, 
#         torch_dtype=torch.float16,
#         device_map=device_map,
#         quantization_config=quantization_config,
        
#     )

#     prompt = "Напиши стих про машинное обучение"
#     messages = [
#         {"role": "system", "content": "Ты T-lite, виртуальный ассистент в Т-Технологии. Твоя задача - быть полезным диалоговым ассистентом."},
#         {"role": "user", "content": prompt}
#     ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=256
#     )
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

#     print(response)
   
# # # ______________________________________
# #     # import os
# #     # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# #     from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# #     # from torch.utils.data import DataLoader, Dataset
# #     import torch.nn as nn
# #     import torch.optim as optim
# #     from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# #     import torch.optim as optim

# #     torch.manual_seed(1942)
# #     tensor = torch.empty((2, 2), dtype=torch.float32).to('meta')

# #     if tensor.is_meta:
# #         # Allocate proper tensor with data before proceeding
# #         tensor = torch.empty((2, 2), dtype=torch.float32)

# #     print(tensor)
# #     model_name = "t-tech/T-lite-it-1.0"
# #     tokenizer = AutoTokenizer.from_pretrained(model_name)
# #     nf4_config = BitsAndBytesConfig (    
# #     load_in_4bit=True,
# #     bnb_4bit_quant_type="nf4",
# #     bnb_4bit_use_double_quant=True,
# #     bnb_4bit_compute_dtype=torch.bfloat16
# #                                     )
    
# # #     quantization_config1 = BitsAndBytesConfig(
# # #    load_in_4bit=True,
# # #    bnb_4bit_compute_dtype=torch.bfloat16,
# # #    bnb_4bit_quant_type="nf4"
# # # )  
# #     quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True, )#llm_int8_enable_fp32_cpu_offload=True, load_in_4bit=True,bnb_4bit_quant_type="bitsandbytes_4bit")#,)
# #     device_map = {  
# # #     "model.embed_tokens.weight": "cpu",
# # #     "model.layers.0.input_layernorm.weight":"cpu",
# # #     "model.layers.0.mlp.down_proj.weight":"cpu",
# # #     "model.layers.0.mlp.gate_proj.weight":"cpu",
# # #     "model.layers.0.mlp.up_proj.weight":"cpu",
# # #     "model.layers.0.post_attention_layernorm.weight":"cpu",
# # #     "model.layers.0.self_attn.k_proj.bias":"cpu",
# # #     "model.layers.0.self_attn.k_proj.weight":"cpu",
# # #     "model.layers.0.self_attn.o_proj.weight":"cpu",
# # #     "model.layers.0.self_attn.q_proj.bias":"cpu",
# # #     "model.layers.0.self_attn.q_proj.weight":"cpu",
# # #     "model.layers.0.self_attn.v_proj.bias":"cpu",
# # #     "model.layers.0.self_attn.v_proj.weight":"cpu",
# # #     "model.layers.1.input_layernorm.weight":"cpu",
# # #     "model.layers.1.mlp.down_proj.weight":"cpu",
# # #     "model.layers.1.mlp.gate_proj.weight":"cpu",
# # #     "model.layers.1.mlp.up_proj.weight":"cpu",
# # #     "model.layers.1.post_attention_layernorm.weight":"cpu",
# # #     "model.layers.1.self_attn.k_proj.bias":"cpu",
# # #     "model.layers.1.self_attn.k_proj.weight":"cpu",
# # #     "model.layers.1.self_attn.o_proj.weight":"cpu",
# # #     "model.layers.1.self_attn.q_proj.bias":"cpu",
# # #     "model.layers.1.self_attn.q_proj.weight":"cpu",
# # #     "model.layers.1.self_attn.v_proj.bias":"cpu",
# # #     "model.layers.1.self_attn.v_proj.weight":"cpu",
# # #     "model.layers.2.input_layernorm.weight":"cpu",
# # #     "model.layers.2.mlp.down_proj.weight":"cpu",
# # #     "model.layers.2.mlp.gate_proj.weight":"cpu",
# # #     "model.layers.2.mlp.up_proj.weight":"cpu",
# # #     "model.layers.2.post_attention_layernorm.weight":"cpu",
# # #     "model.layers.2.self_attn.k_proj.bias":"cpu",
# # #     "model.layers.2.self_attn.k_proj.weight":"cpu",
# # #     "model.layers.2.self_attn.o_proj.weight":"cpu",
# # #     "model.layers.2.self_attn.q_proj.bias":"cpu",
# # #     "model.layers.2.self_attn.q_proj.weight":"cpu",
# # #     "model.layers.2.self_attn.v_proj.bias":"cpu",
# # #     "model.layers.2.self_attn.v_proj.weight":"cpu",
# # #     "model.layers.3.input_layernorm.weight":"cpu",
# # #     "model.layers.3.mlp.down_proj.weight":"cpu",
# # #     "model.layers.3.mlp.gate_proj.weight":"cpu",
# # #     "model.layers.3.mlp.up_proj.weight":"cpu",
    
# #     "transformer.word_embeddings": 0,  
# #     "transformer.word_embeddings_layernorm": 0,  
# #     "lm_head": "cpu",  
# #     "transformer.h": 0,  
# #     "transformer.ln_f": 0 

# # }  
# # #     loader = DataLoader(
# # #     dataset,
# # #     batch_size=1,
# # #     collate_fn=collate_wrapper,
# # #     pin_memory=True
# # # ) 
# #     print("trur or false ----------------------------?", torch.cuda.is_available())
# #     # train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
# #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
# #     model = AutoModelForCausalLM.from_pretrained(
# #         model_name,
# #         # torch_dtype=torch.bfloat16,
# #         quantization_config=quantization_config,
# #         low_cpu_mem_usage=True,
# #         # quantization_config=nf4_config,
# #         # torch_dtype="auto",
# #         # device_map=device_map,
# #         device_map="auto",
# #         torch_dtype=torch.float16,
# #         # device_map = "balanced",
# #         # device_map="cuda",
# #         # load_in_8bit = True,
        
   
# #     )
# #     # model.to_empty(device) 
# #     # from accelerate import disk_offload  
# #     # disk_offload(model=model, offload_dir="offload")
# #     # # Example for checking meta tensor
# #     # tensor = torch.empty((2, 2), dtype=torch.float32).to('meta')

# #     # if tensor.is_meta:
# #     # # Allocate proper tensor with data before proceeding
# #     #     tensor = torch.empty((2, 2), dtype=torch.float32)

# #     # print(tensor)

# # # # Iterate over the DataLoader
# # #     print(f"Dataset size: {len(dataset)}")
# # #     print(f"DataLoader batch size: {train_loader.batch_size}")

# # #     for epoch in range(1): # Example for one epoch
# # #         print(f"\n--- Epoch {epoch+1} ---")
# # #         for i, batch in enumerate(train_loader):
# # #             # DataLoader yields batches. Each 'batch' is typically a tuple or list
# # #             # containing tensors for features and labels.
# # #             features, labels = batch
# # #             print(f"Batch {i+1}: Features shape={features.shape}, Labels shape={labels.shape}")
# # #             # Here you would typically perform your training steps:
# # #             # model.train()
# # #             # optimizer.zero_grad()
# # #             # outputs = model(features)
# # #             # loss = criterion(outputs, labels)
# # #             # loss.backward()
# # #             # optimizer.step()


# #     # model = FSDP(model)
# #     # optimizer = optim.Adam(model.parameters())
# #     # optim.SGD(model, optimizer)

# #     # optimizer = optim.SGD(model.parameters(), lr=0.05)
# #     # loss_fn = nn.MSELoss()

# #     # dataset = Dataset()
# #     # dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)#, shuffle=True, pin_memory=True)
    
# #     # dataloader = DataLoader(dataset, batch_size=32)
# #     # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# #     # optimizer.zero_grad()
# #     # for data in dataloader:
# #     #     inputs, labels = data
# #     #     outputs = model(inputs)
# #     #     loss = criterion(outputs, labels)
# #     #     loss.backward()
        
# #     #     if (i+1) % accumulation_steps == 0:
# #     #         optimizer.step()
# #     #         optimizer.zero_grad()
# # # _____________________________________
   
# # # __________________________________________________
# #     # from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# #     # torch.manual_seed(42)
# #     # from accelerate import infer_auto_device_map

# #     # # sysctl -w vm.swappiness=1
# #     # # sudo sysctl -w vm.max_map_count=262144
# #     # model_name = "t-tech/T-lite-it-1.0"
# #     # tokenizer = AutoTokenizer.from_pretrained(model_name)
    
# #     # model = AutoModelForCausalLM.from_pretrained(
# #     #     model_name,
# #     #     # torch_dtype="auto",
# #     #     # device_map = infer_auto_device_map(model, max_memory={0: "10GiB", 1: "10GiB", "cpu": "30GiB"}),
# #     #     # device_map="auto",
# #     #     # load_in_4bit=True,
# #     #     # bnb_4bit_use_double_quant=True,
# #     #     # load_in_8bit = True,
# #     #     device_map="auto",
# #     #     # load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
# #     #     quantization_config = BitsAndBytesConfig(
# #     #         llm_int8_enable_fp32_cpu_offload=True, load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
# #     #         #   load_in_4bit=True, bnb_4bit_quant_type="nf4",
# #     #         ),
# #     #     torch_dtype=torch.bfloat16,
# #     #     trust_remote_code=True,
# #     #     # quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
# #     # )
# # # ________________________________________________________
# #     # prompt = "Напиши стих про машинное обучение"
# #     # messages = [
# #     #     {"role": "system", "content": "Ты T-lite, виртуальный ассистент в Т-Технологии. Твоя задача - быть полезным диалоговым ассистентом."},
# #     #     {"role": "user", "content": prompt}
# #     # ]
# #     # text = tokenizer.apply_chat_template(
# #     #     messages,
# #     #     tokenize=False,
# #     #     add_generation_prompt=True
# #     # )
# #     # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# #     # generated_ids = model.generate(
# #     #     **model_inputs,
# #     #     max_new_tokens=256
# #     # )
# #     # generated_ids = [
# #     #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# #     # ]

# #     # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# #     # print(response)
# #     from huggingface_hub import model_info  
  
# #     # Получить информацию о модели OpenAI Whisper Large V3:  
# #     m = model_info("openai/whisper-large-v3")  
# #     # Распечатать параметры модели и общее использование памяти:  
# #     print(m.safetensors)  


# if __name__ == "__main__":  
#     main()  