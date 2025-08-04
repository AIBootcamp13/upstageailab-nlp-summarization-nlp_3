import pandas as pd
import sys
import torch

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.append('.')    

from src.data.dialogue_paraphraser import DialogueParaphraser
from src.data.simple_dataset import SimpleDataset

train_path = r"/data/ephemeral/home/work/python/upstageailab-nlp-summarization-nlp_3/data/raw/train.csv"

train = pd.read_csv(train_path, encoding="utf-8")  # or cp949 if needed


try:
    # 1. 클래스 인스턴스 생성
    final_paraphraser = DialogueParaphraser()
    tokenizer = final_paraphraser.paraphraser.tokenizer

    # 1. 유효한 데이터만 필터링
    valid_mask = train['dialogue'].apply( lambda x: len(tokenizer.encode(x)) < 512)
    valid_df = train[valid_mask]
    print(f"유효한 샘플 수: {len(valid_df)}")

    # 2. 최소한의 Dataset 래퍼 사용
    dataset = SimpleDataset(valid_df)
    
    # 3. DataLoader 생성
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    new_data = []
    
    # 4. 배치 처리
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="배치 처리")):
    
        dialogues  = batch['dialogue']      # 이미 리스트 형태
        summaries  = batch['summary']
        fnames     = batch['fname']
        topics     = batch['topic']

        try:
            # 배치 처리
            """ paraphrased_batch = []
            similarities_batch = []
            for dialogue in dialogues:
                paraphrased = final_paraphraser.paraphrase_dialogue(dialogue)
                similarity = final_paraphraser.calculate_semantic_similarity(dialogue, paraphrased)
                paraphrased_batch.append(paraphrased)
                similarities_batch.append(similarity) """
            paraphrased_batch = final_paraphraser.paraphrase_dialogue_batch(dialogues)
            similarities_batch = final_paraphraser.calculate_semantic_similarity_batch(dialogues, paraphrased_batch)

            
            # 결과 처리
            for dialogue, paraphrased, similarity, summary, fname, topic in zip(
                dialogues, paraphrased_batch, similarities_batch, summaries, fnames, topics
            ):
                if similarity >= 0.85:
                    new_data.append({
                        'dialogue': paraphrased,
                        'summary': summary,
                        'fname': f'train_aug_{fname}',
                        'topic': topic,
                        'similarity': similarity
                    })
                else:
                    print(f"{fname} similarity: {similarity:.3f} skipped", flush=True)

        except Exception as e:
            print(f"배치 {batch_idx} 처리 중 오류: {e}", flush=True)
            continue
        
        if batch_idx % 10 == 0:
            print(f"\n배치 {batch_idx} 샘플:", flush=True)
            print(f"원본 개수: {len(dialogues)}", flush=True)
            print(f"생성 개수: {len(paraphrased_batch)}", flush=True)
            if len(paraphrased_batch) > 0:
                print(f"첫 번째 원본: {dialogues[0]}...", flush=True)
                print(f"첫 번째 생성: {paraphrased_batch[0]}...", flush=True)
            print(f"배치 {batch_idx} 완료, 총 생성된 데이터: {len(new_data)}", flush=True)
        
        torch.cuda.empty_cache()            
    
    # 결과 저장
    result = pd.concat([train, pd.DataFrame(new_data)], ignore_index=True)
    result.to_csv("./data/raw/new_train.csv", index=False)
    print(f"총 {len(new_data)}건 추가 완료")          

except Exception as e:
    print(f"최종 실행 중 오류 발생: {e}")

# nohup poetry run python src/make-data.py &    