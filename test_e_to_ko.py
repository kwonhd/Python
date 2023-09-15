import fitz
import requests
import time
import re
import json

# English to Korean Translation Func.
def papagoAPI(client_id, client_secret, sentence):
    request_url = "https://openapi.naver.com/v1/papago/n2mt"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    params = {"source": "en", "target": "ko", "text": sentence}
    
    # Encode the headers and data as UTF-8
    headers_encoded = {key: value.encode('utf-8') if isinstance(value, str) else value for key, value in headers.items()}
    data_encoded = {key: value.encode('utf-8') if isinstance(value, str) else value for key, value in params.items()}
    
    response = requests.post(request_url, headers=headers_encoded, data=data_encoded)
    
    # Check if the response is successful (status code 200)
    if response.status_code == 200:
        result = json.loads(response.text)
        
        # Check if the response contains the expected structure
        if "message" in result and "result" in result["message"] and "translatedText" in result["message"]["result"]:
            return result["message"]["result"]["translatedText"]
        else:
            print("Error: Unexpected response format from Papago API.")
            return None
    else:
        print(f"Error: Request failed with status code {response.status_code}")
        return None



# English PDF file to 한국어 Text 파일 저장 및 읽기
def pdfToText(inputFile):
    doc = fitz.open(inputFile)
    print("문서 페이지 수: ", len(doc))

    # pdf -> Text save.
    ext_text = ""
    for page in doc:
        temp = page.get_text()
        temp = temp.replace("\n", " ")
        ext_text += temp
    print("Text length : ", len(ext_text))

    # Find sentences with a period and create paragraphs
    txt = ""
    final_sent = re.compile("[^.]*\.")
    sentences = final_sent.findall(ext_text)

    # Combine sentences into paragraphs
    paragraph = ""
    for sentence in sentences:
        paragraph += sentence.strip() + " "
        if len(paragraph) > 500:
            txt += paragraph + "\n"
            paragraph = ""

    if paragraph:
        txt += paragraph + "\n"

    # Save extracted text as a TXT file
    with open('./data/ext_text.txt', 'w', encoding='utf-8') as file:
        file.write(txt)

    # Read the saved TXT file
    with open('./data/ext_text.txt', 'r', encoding='utf-8') as file:
        text = file.readlines()

    print("문장 길이 : ", len(text))
    return text

# Papago API 번역 및 저장
def trans(client_id, client_secret, input_text, line=10):
    text = ""
    for i in range(min(line, len(input_text))):
        result_txt = papagoAPI(client_id, client_secret, input_text[i])
        if result_txt is not None:
            print("번역결과 {} : {}".format(i, result_txt))
            text += result_txt + "\n"
        else:
            print("번역 결과가 없습니다.")
        time.sleep(2)

    trans_file = open("./data/trans_result.txt", 'w', encoding='utf-8')
    trans_file.write(text)
    trans_file.close()


if __name__ == "__main__":
    # pdf to text 함수 호출
    eng_text = pdfToText('./data/test.pdf')
    
    # Naver 번역 API ID, Secret(password)
    id = "발급받은ClientID"
    secret = "발급받은ClientSecret"
    
    # Text 영문, Papago 번역 후 Text 파일 저장
    trans(id, secret, eng_text, line=2)

