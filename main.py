import streamlit as st
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


# Streamlit 앱 제목
st.title("자비스")

# 모델과 토크나이저 로드 (사전에 학습된 모델 경로로 변경)
model_path = './fine_tuned_albert'

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

#model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
#tokenizer = AutoTokenizer.from_pretrained(model_path)

#고유한 답변리스트
with open('unique_answers.pkl', 'rb') as f:
    unique_answers = pickle.load(f)

# 사용자 입력 받기
user_input = st.text_input("안녕하세요 반갑습니다. 무엇을 도와드릴까요?")

# 예측 버튼
if st.button('예측'):
    if user_input:
        # 입력된 질문을 토큰화
        inputs = tokenizer(user_input, return_tensors="pt")

        # 모델 예측
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1) # 예측된 클래스 번호
            
        #최저확률 임계치
        max_prob = torch.max(probabilities).item()
        print(max_prob)
        if max_prob < 0.2:
            st.write("정확하지만 답변해볼게요")
        else:
            predicted_answer = unique_answers[predicted_class]
            st.write(f"예측된 답변: {predicted_answer}")

        
        

        # with torch.no_grad():
        #    generated_outputs = model.generate(inputs['input_ids'], max_length=50, num_beams=5, early_stopping=True)
        #    predicted_answer = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)

        # 예측된 클래스 번호에 해당하는 답변 출력
        if predicted_class < len(unique_answers):
            predicted_answer = unique_answers[predicted_class]  # 고유 답변 리스트에서 매핑
            st.write(f"예측된 답변: {predicted_answer}")
        else:
            st.write("해당하는 답변이 없습니다.")
    else:
        st.write("질문을 입력하세요.")
