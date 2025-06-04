from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# 加载预训练模型和分词器
model = AutoModelForSequenceClassification.from_pretrained("left0ver/bert-base-chinese-finetune-sentiment-classification", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 设置为评估模式

def predict_sentiment_batch(texts, ids, batch_size=16):
    """批量处理文本的情感分析"""
    results = []
    
    # 分批处理
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        # 对批次进行编码
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 批量预测
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
        
        # 处理批次结果
        label_map = {0: "负面", 1: "正面"}
        for j, (pred, id_val) in enumerate(zip(predictions, batch_ids)):
            confidence = probabilities[j][pred].item()
            
            result = {
                "id": id_val,
                "sentiment": int(pred),
                "confidence": confidence,
                "prob_positive": probabilities[j][1].item(),
                "prob_negative": probabilities[j][0].item(),
                "sentiment_label": label_map[pred],
            }
            results.append(result)
    
    return pd.DataFrame(results)

def process_comments(save_every=1000, batch_size=16):
    """处理评论并定期保存结果"""
    # 加载数据
    df = pd.read_csv('data/short_coms_all.csv', encoding='utf-8-sig')
    
    # 加载已处理的结果或创建新的结果DataFrame
    result_file = 'data/short_coms_sentiment.csv'
    try:
        df_res = pd.read_csv(result_file, encoding='utf-8-sig')
        processed_ids = set(df_res['id'].unique())
        print(f"已加载 {len(df_res)} 条处理过的评论")
        # 结果文件已存在，我们只需追加新结果
        file_exists = True
    except FileNotFoundError:
        df_res = pd.DataFrame(columns=['id', 'sentiment', 'confidence', 
                                      'prob_positive', 'prob_negative', 'sentiment_label'])
        processed_ids = set()
        print("创建新的结果文件")
        file_exists = False
    
    # 过滤掉已经处理过的评论
    df = df[~df['review_id'].isin(processed_ids)]
    print(f"待处理评论数量: {len(df)}")
    
    if len(df) == 0:
        print("所有评论已处理完毕")
        return
    
    # 处理评论
    texts = []
    ids = []
    processed_count = 0
    batch_results = pd.DataFrame()
    
    with tqdm(total=len(df), desc="处理评论") as pbar:
        for index, row in df.iterrows():
            text = row['review_content']
            id_val = row['review_id']
            
            # 跳过空评论
            if pd.isna(text) or text.strip() == "":
                pbar.update(1)
                continue
            
            texts.append(text)
            ids.append(id_val)
            
            # 达到批次大小时进行批处理
            if len(texts) >= batch_size:
                results = predict_sentiment_batch(texts, ids, batch_size)
                batch_results = pd.concat([batch_results, results], ignore_index=True)
                
                processed_count += len(texts)
                texts = []
                ids = []
                
                # 每处理指定数量的评论保存一次
                if processed_count % save_every < batch_size:
                    # 追加保存到文件
                    batch_results.to_csv(result_file, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
                    file_exists = True
                    # print(f"\n已处理并保存 {processed_count} 条评论")
                    # 清空批次结果，减少内存占用
                    batch_results = pd.DataFrame()
            
            pbar.update(1)
        
        # 处理剩余的评论
        if texts:
            results = predict_sentiment_batch(texts, ids, batch_size)
            batch_results = pd.concat([batch_results, results], ignore_index=True)
            processed_count += len(texts)
    
    # 保存最终结果
    if not batch_results.empty:
        batch_results.to_csv(result_file, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
    print(f"\n已完成所有处理，共 {processed_count} 条评论")

# 使用示例
if __name__ == "__main__":
    # 定义批处理大小（可根据GPU内存调整）
    BATCH_SIZE = 16  # 较大的批次更有效利用GPU
    SAVE_EVERY = 1000  # 每处理1000条保存一次
    
    process_comments(save_every=SAVE_EVERY, batch_size=BATCH_SIZE)