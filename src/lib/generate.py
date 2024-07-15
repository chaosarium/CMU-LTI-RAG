import os
from langchain_core.documents import Document
from langchain.output_parsers.regex import RegexParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.chains.combine_documents.map_rerank import MapRerankDocumentsChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_together import Together
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompt_values import ChatPromptValue
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
load_dotenv()

# prompt engineering references
# https://www.promptingguide.ai/
# https://www.pinecone.io/learn/series/langchain/langchain-prompt-templates/


def format_documents(docs: list[Document]):
    res = ['']
    for doc in docs:
        res.append(doc.page_content)
    res.append('')
    
    return "\n-----\n".join(res)

def tag_message(role: str, style: str, pos: str):
    match (role, style, pos):
        case ('system', 'llama', 'start'):
            return f'<<SYS>>'
        case ('system', 'llama', 'end'):
            return f'<</SYS>>'
        case ('human', 'llama', 'start'):
            return f''
        case ('human', 'llama', 'end'):
            return f''
        case ('whole', 'llama', 'start'):
            return f'[INST] '
        case ('whole', 'llama', 'end'):
            return f' [/INST]'
        
        case ('system', 'gemma', 'start'):
            return f'System: '
        case ('system', 'gemma', 'end'):
            return f''
        case ('human', 'gemma', 'start'):
            return f'Human: '
        case ('human', 'gemma', 'end'):
            return f''
        case ('whole', 'gemma', 'start'):
            return f''
        case ('whole', 'gemma', 'end'):
            return f''
        
        case ('system', 'mixtral', 'start'):
            return f''
        case ('system', 'mixtral', 'end'):
            return f''
        case ('human', 'mixtral', 'start'):
            return f''
        case ('human', 'mixtral', 'end'):
            return f''
        case ('whole', 'mixtral', 'start'):
            return f'[INST]'
        case ('whole', 'mixtral', 'end'):
            return f'[/INST]'
        case _:
            raise ValueError

def mk_context_qa_prompt(question: str, documents: list[Document], style: str) -> str:
    return f"""{tag_message('whole', style, 'start')}{tag_message('system', style, 'start')}### Instructions ###
You are a question-answering assistant at Carnegie Mellon University (CMU). You are tasked with answering factual questions CMU or the Language Technologies Institute (LTI) at CMU. Use the following documents as context to answer the question in one sentence. You are looking for the answer to the question "{question}".

### Examples ###

User: Who taught the course 51-425 Design Center: Beginning Book Arts Lab in the semester Fall 2023?
Response: Joseph Dicey.
User: Which model did Fatemehsadat Mireshghallah find in their paper to have an AUC of 0.81?
Response: OPT-125M.
User: Who is sponsoring the event Carnival Activities Tent at CMU's Spring Carnival 2024?
Response: The Spring Carnival Committee and the CMU Alumni Association.
User: What are Graham Neubig's research interests?
Response: Machine Translation, Natural Language Processing, Spoken Language Processing, Machine Learning.{tag_message('system', style, 'end')}

{tag_message('human', style, 'start')}### Context ###
{format_documents(documents)}
### Response ###

User: {question}
Response:{tag_message('human', style, 'end')}{tag_message('whole', style, 'end')}
"""

def mk_no_context_qa_prompt(question: str, style: str) -> str:
    return f"""{tag_message('whole', style, 'start')}{tag_message('system', style, 'start')}You are a question-answering assistant at Carnegie Mellon University (CMU). Answer the user's question{tag_message('system', style, 'end')}
{tag_message('human', style, 'start')}
User: {question}
Response:{tag_message('human', style, 'end')}{tag_message('whole', style, 'end')}
"""

def mk_hypothetical_doc_prompt(question: str, style: str) -> str:
    def word_and_example(q):
        ql = q.lower()
        if "course" in ql or "teaching" in ql or "taught" in ql or "instructor" in ql:
            word = "course description"
            example = '''Question: Who taught 42-202 Physiology in Fall 2023?
Search Result:
Semester: Fall 2023 (aka F23)
Course Name: Physiology
Course Number: 42-202
Department: Biomedical Engineering
Number of Units: 9
Prerequisites: 03-121 or 03-151
Instructors: Phil Campbell
Rooms: POS 151
Locations: Pittsburgh, Pennsylvania'''
        elif "position" in ql or "professional title" in ql or "interests" in ql or "email" in ql or "office" in ql or "phone number" in ql:
            word = "faculty profile"
            example = '''Question: What is Graham Neubig's position at CMU?
Search Result:
Name: Graham Neubig
Title: Associate Professor
Email gneubig@cs.cmu.edu
Phone: 
Office: 5409 Gates & Hillman Centers
Interests: Machine Translation, Natural Language Processing, Spoken Language Processing, Machine Learning'''
        elif "paper" in ql or "author" in ql or "published" in ql:
          word = "set of paper metadata"
          example = '''Question: Who is the first author of the paper titled "Text Matching Improves Sequential Recommendation by Reducing Popularity Biases"?
Search Result:
publication venue: International Conference on Information and Knowledge Management
title: Text Matching Improves Sequential Recommendation by Reducing Popularity Biases
authors: Zhenghao Liu, Senkun Mei, Chenyan Xiong, Xiaohua Li, Shi Yu, Zhiyuan Liu, Yu Gu, Ge Yu
year: 2023
tldr: None
abstract: This paper proposes Text mAtching based SequenTial rEcommenda-tion model (TASTE), which maps items and users in an embedding space and recommends items by matching their text representations. TASTE verbalizes items and user-item interactions using identifiers and attributes of items. To better characterize user behaviors, TASTE additionally proposes an attention sparsity method, which enables TASTE to model longer user-item interactions by reducing the self-attention computations during encoding. Our experiments show that TASTE outperforms the state-of-the-art methods on widely used sequential recommendation datasets. TASTE alleviates the cold start problem by representing long-tail items using full-text modeling and bringing the benefits of pretrained language models to recommendation systems. Our further analyses illustrate that TASTE significantly improves the recommendation accuracy by reducing the popularity bias of previous item id based recommendation models and returning more appropriate and text-relevant items to satisfy users. All codes are available at https://github.com/OpenMatch/TASTE.
'''
        elif "when is" in ql  or "event" in ql  or "happening" in ql or "occurring" in ql or "hosted" in ql or "what time" in ql:
            word = "event description"
            example = '''Question: On what date is the 25th Reunion Dinner Reception (Class of 1999) happening?
Title: 25th Reunion Dinner Reception (Class of 1999)
Date: April 13, 2024
Time: 6:00 PM-9:00 PM ET

Celebrate your reunion in the beautiful College of Fine Arts! Enjoy a variety of food stations, beverages and desserts and toast to an amazing 25 years with fellow classmates from the Class of 1999! Note: Registration required. This event is open to the Class of 1999 and their guests only.CostAdultNow through Feb. 23: $35 per person 12 years of age and older.Feb. 24 through Apr. 5: $45 per person 12 years of age and older.ChildNow through Feb. 23: $15 per person ages 6-12.Feb. 24 through Apr. 5: $20 per person ages 6-12.
'''
        elif "director" in ql or "program" in ql or "requirement" in ql or "degree" in ql or "required" in ql:
            word = "program description"
            example = '''Question: Who is the Associate Dean for Master’s Programs at CMU?
Handbook-MSAII-2022-2023
7 School and Departmental Information 
The following are key personnel with whom you may  interact during your time at Carnegie 
Mellon: 
 Martial Hebert     
 Dean, School of Computer Science 
 University Professor 
 GHC 6105  412-268-5704 
 
hebert@cs.cmu.edu   
https://www.cs.cmu.edu/~hebert/   
 
David Garlan Associate Dean for Master’s Programs, SCS Professor TCS 420 
garlan@cs.cmu.edu  
http://www.cs.cmu.edu/~garlan/academics.htm   
 
Carolyn  Rosé      
 Interim Director, Language Technologies Institute  Professor  GHC 5415 
 412-268-7130 
 cprose@cs.cmu.edu
'''
        else:
            word = "webpage"
            example = '''Question: What is the name of CMU's mascot?
Search Result:
About Scotty, CMU's mascot:
The Scottish terrier has long been a familiar figure around Carnegie Mellon's campus. For years students have suited up in an unofficial Scottish terrier costume to excite the fans at athletic events. But the relationship between the Scottish terrier breed and Carnegie Mellon far precedes anybody doing somersaults in a dog costume. Andrew Carnegie, founder of the university, kept a Scottish terrier as his pet.

Scotty's road from popular icon to official mascot of the university began in 2006. Carnegie Mellon formed a Mascot Identity Task Force in November 2006, which consisted of students, faculty, staff and alumni. The Task Force was co-chaired by Director of Athletics Susan Bassett and Dean of Student Affairs Jennifer Church.  
  
The mascot selection process included a series of surveys and a university Town Hall meeting. Nearly 78 percent of 2,370 students surveyed in February 2007 voted for the Scottish terrier, and approximately 25 percent of 400 alumni surveyed thought the Scottish terrier was already the mascot.  
  
In the spring, the Task Force partnered with SME Branding — a firm with more than 17 years of experience creating mascots for professional sports teams and universities — to develop the graphics for the mascot. During October, students and alumni reviewed potential mascot images in focus groups.  
  
Carnegie Mellon's official mascot debuted at the Nov. 10, 2007 home football game. The graphic features a profile of a distinguished, bold Scottish terrier sporting a plaid scarf around his neck. The dog is contained in a shield, representing Carnegie Mellon's Scottish heritage.  
  
The Task Force then partnered with a mascot costume company to design our Scottish terrier in the winter of 2007. The official Scotty costume was unveiled at the 2008 Spring Carnival.

Fun Fact about Scotty, CMU's mascot:
The Scottish terrier breed is known for its keen, alert and intelligent expression. Its temperament is described as determined and thoughtful while its physical aspects exemplify strength, power and agility in a small package. Many of these traits are also apparent throughout the university, making the Scottish terrier a natural choice for Carnegie Mellon's mascot.
'''
        return word, example
    specdocword, specexample = word_and_example(question)

    return f"""{tag_message('whole', style, 'start')}{tag_message('system', style, 'start')}### Instructions ###
You are a search engine at Carnegie Mellon University (CMU). Produce a search result that answers the user's question.{tag_message('system', style, 'end')}

{tag_message('human', style, 'start')}Generate a {specdocword} that could answer the question "{question}". Give your response in the same format as the example below. Only generate the part after 'Search Result'; do not repeat the question or the words 'Search Result'.

### Example ###

{specexample}

### Your turn. Produce a search result. ###

Question: {question}
Search Result:
{tag_message('human', style, 'end')}{tag_message('whole', style, 'end')}
"""

def mk_hypothetical_doc_prompt(question: str, style: str) -> str:
    def word_and_example(q):
        ql = q.lower()
        if "course" in ql or "teaching" in ql \
        or "taught" in ql or "instructor" in ql:
            word = "course description"
            example = '''Question: Who taught 42-202 Physiology in Fall 2023?
Search Result:
Semester: Fall 2023 (aka F23)
Course Name: Physiology
Course Number: 42-202
Department: Biomedical Engineering
Number of Units: 9
Prerequisites: 03-121 or 03-151
Instructors: Phil Campbell
Rooms: POS 151
Locations: Pittsburgh, Pennsylvania'''
        elif "position" in ql\
          or "professional title" in ql\
          or "interests" in ql\
          or "email" in ql\
          or "office" in ql\
          or "phone number" in ql:
            word = "faculty profile"
            example = '''Question: What is Graham Neubig's position at CMU?
Search Result:
Name: Graham Neubig
Title: Associate Professor
Email gneubig@cs.cmu.edu
Phone: 
Office: 5409 Gates & Hillman Centers
Interests: Machine Translation, Natural Language Processing, Spoken Language Processing, Machine Learning'''
        elif "paper" in ql\
          or "author" in ql\
          or "published" in ql:
          word = "set of paper metadata"
          example = '''Question: Who is the first author of the paper titled "Text Matching Improves Sequential Recommendation by Reducing Popularity Biases"?
Search Result:
publication venue: International Conference on Information and Knowledge Management
title: Text Matching Improves Sequential Recommendation by Reducing Popularity Biases
authors: Zhenghao Liu, Senkun Mei, Chenyan Xiong, Xiaohua Li, Shi Yu, Zhiyuan Liu, Yu Gu, Ge Yu
year: 2023
tldr: None
abstract: This paper proposes Text mAtching based SequenTial rEcommenda-tion model (TASTE), which maps items and users in an embedding space and recommends items by matching their text representations. TASTE verbalizes items and user-item interactions using identifiers and attributes of items. To better characterize user behaviors, TASTE additionally proposes an attention sparsity method, which enables TASTE to model longer user-item interactions by reducing the self-attention computations during encoding. Our experiments show that TASTE outperforms the state-of-the-art methods on widely used sequential recommendation datasets. TASTE alleviates the cold start problem by representing long-tail items using full-text modeling and bringing the benefits of pretrained language models to recommendation systems. Our further analyses illustrate that TASTE significantly improves the recommendation accuracy by reducing the popularity bias of previous item id based recommendation models and returning more appropriate and text-relevant items to satisfy users. All codes are available at https://github.com/OpenMatch/TASTE.
'''
        elif "when is" in ql \
          or "event" in ql \
          or "happening" in ql\
          or "occurring" in ql\
          or "hosted" in ql\
          or "what time" in ql:
            word = "event description"
            example = '''Question: On what date is the 25th Reunion Dinner Reception (Class of 1999) happening?
Title: 25th Reunion Dinner Reception (Class of 1999)
Date: April 13, 2024
Time: 6:00 PM-9:00 PM ET

Celebrate your reunion in the beautiful College of Fine Arts! Enjoy a variety of food stations, beverages and desserts and toast to an amazing 25 years with fellow classmates from the Class of 1999! Note: Registration required. This event is open to the Class of 1999 and their guests only.CostAdultNow through Feb. 23: $35 per person 12 years of age and older.Feb. 24 through Apr. 5: $45 per person 12 years of age and older.ChildNow through Feb. 23: $15 per person ages 6-12.Feb. 24 through Apr. 5: $20 per person ages 6-12.
'''
        elif "director" in ql or "program" in ql \
          or "requirement" in ql or "degree" in ql:
            word = "program description"
            example = '''Question: Who is the Associate Dean for Master’s Programs at CMU?
Handbook-MSAII-2022-2023
7 School and Departmental Information 
The following are key personnel with whom you may  interact during your time at Carnegie 
Mellon: 
 Martial Hebert     
 Dean, School of Computer Science 
 University Professor 
 GHC 6105  412-268-5704 
 
hebert@cs.cmu.edu   
https://www.cs.cmu.edu/~hebert/   
 
David Garlan Associate Dean for Master’s Programs, SCS Professor TCS 420 
garlan@cs.cmu.edu  
http://www.cs.cmu.edu/~garlan/academics.htm   
 
Carolyn  Rosé      
 Interim Director, Language Technologies Institute  Professor  GHC 5415 
 412-268-7130 
 cprose@cs.cmu.edu
'''
        else:
            word = "webpage"
            example = '''Question: What is the name of CMU's mascot?
Search Result:
About Scotty, CMU's mascot:
The Scottish terrier has long been a familiar figure around Carnegie Mellon's campus. For years students have suited up in an unofficial Scottish terrier costume to excite the fans at athletic events. But the relationship between the Scottish terrier breed and Carnegie Mellon far precedes anybody doing somersaults in a dog costume. Andrew Carnegie, founder of the university, kept a Scottish terrier as his pet.

Scotty's road from popular icon to official mascot of the university began in 2006. Carnegie Mellon formed a Mascot Identity Task Force in November 2006, which consisted of students, faculty, staff and alumni. The Task Force was co-chaired by Director of Athletics Susan Bassett and Dean of Student Affairs Jennifer Church.  
  
The mascot selection process included a series of surveys and a university Town Hall meeting. Nearly 78 percent of 2,370 students surveyed in February 2007 voted for the Scottish terrier, and approximately 25 percent of 400 alumni surveyed thought the Scottish terrier was already the mascot.  
  
In the spring, the Task Force partnered with SME Branding — a firm with more than 17 years of experience creating mascots for professional sports teams and universities — to develop the graphics for the mascot. During October, students and alumni reviewed potential mascot images in focus groups.  
  
Carnegie Mellon's official mascot debuted at the Nov. 10, 2007 home football game. The graphic features a profile of a distinguished, bold Scottish terrier sporting a plaid scarf around his neck. The dog is contained in a shield, representing Carnegie Mellon's Scottish heritage.  
  
The Task Force then partnered with a mascot costume company to design our Scottish terrier in the winter of 2007. The official Scotty costume was unveiled at the 2008 Spring Carnival.

Fun Fact about Scotty, CMU's mascot:
The Scottish terrier breed is known for its keen, alert and intelligent expression. Its temperament is described as determined and thoughtful while its physical aspects exemplify strength, power and agility in a small package. Many of these traits are also apparent throughout the university, making the Scottish terrier a natural choice for Carnegie Mellon's mascot.
'''
        return word, example
    specdocword, specexample = word_and_example(question)

    return f"""{tag_message('whole', style, 'start')}{tag_message('system', style, 'start')}### Instructions ###
You are a search engine at Carnegie Mellon University (CMU). Produce a search result that answers the user's question.{tag_message('system', style, 'end')}

{tag_message('human', style, 'start')}Generate a {specdocword} that could answer the question "{question}". Give your response in the same format as the examples below.

### Example ###

{specexample}

### Your turn. Produce a search result. ###

Question: {question}
Search Result:
{tag_message('human', style, 'end')}{tag_message('whole', style, 'end')}
"""

def extract_answer(question: str, llm_output: str):
    
    llm_output.replace(question, "<question>")
    
    # some cleaning
    if 'Response:' in llm_output:
        llm_output = ''.join(llm_output.split('Response:')[1:])
        llm_output = llm_output.strip()
    elif 'is:\n' in llm_output:
        llm_output = ''.join(llm_output.split('is:\n')[1:])
        llm_output = llm_output.strip()
    elif 'include:\n' in llm_output:
        llm_output = ''.join(llm_output.split('include:\n')[1:])
        llm_output = llm_output.strip()
    
    return llm_output


class GeneratorBase:
    def __init__(self) -> None:
        pass
    
    def __call__(self, prompt: str) -> str:
        raise NotImplementedError
    
    def answer(self, query: str, context: list[Document]) -> str:
        raise NotImplementedError

class OllamaGeneratorBase(GeneratorBase):
    def __init__(self) -> None:
        super().__init__()
        self.model_name: str
        self.prompt_style: str

    def __call__(self, prompt: str | ChatPromptValue, max_tokens=128, temperature=0.7, top_k=1) -> str:

        llm = ChatOllama(
            model=self.model_name,
            base_url="",
            temperature=temperature,
            num_predict=max_tokens,
            top_k=top_k,
        )
        return llm.invoke(prompt).content
    
    def augment_query(self, question: str) -> tuple[str, str]:
        prompt = mk_hypothetical_doc_prompt(question,self.prompt_style )
        output = self(prompt, temperature=0.9, top_k=50, max_tokens=128)
        return output, prompt

    def no_context_answer(self, question: str) -> tuple[str, str, str]:
        prompt = mk_no_context_qa_prompt(question, self.prompt_style)
        output: str = self(prompt, max_tokens=96, temperature=0, top_k=1)
        return extract_answer(question, output), output, prompt

    def answer_with_context(self, question: str, documents: list[Document]) -> tuple[str, str, str]:
        prompt = mk_context_qa_prompt(question, documents, self.prompt_style)
        output: str = self(prompt, max_tokens=96, temperature=0, top_k=1)
        return extract_answer(question, output), output, prompt

class TogetherGeneratorBase(GeneratorBase):
    def __init__(self) -> None:
        super().__init__()
        self.model_name: str
        self.prompt_style: str
        self.together_api_key: str

    def __call__(self, prompt: str | ChatPromptValue, max_tokens=128, temperature=0.7, top_k=1) -> str:
        llm = Together(
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            together_api_key=self.together_api_key
        )
        return llm.invoke(prompt)
    
    def augment_query(self, question: str) -> tuple[str, str]:
        prompt = mk_hypothetical_doc_prompt(question,self.prompt_style )
        output = self(prompt, temperature=0.9, top_k=50, max_tokens=128)
        return output, prompt

    def no_context_answer(self, question: str) -> tuple[str, str, str]:
        prompt = mk_no_context_qa_prompt(question, self.prompt_style)
        output: str = self(prompt, max_tokens=96, temperature=0, top_k=1)
        return extract_answer(question, output), output, prompt

    def answer_with_context(self, question: str, documents: list[Document]) -> tuple[str, str, str]:
        prompt = mk_context_qa_prompt(question, documents, self.prompt_style)
        output: str = self(prompt, max_tokens=96, temperature=0, top_k=1)
        return extract_answer(question, output), output, prompt

class Llama13BGenerator(TogetherGeneratorBase):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "meta-llama/Llama-2-13b-chat-hf"
        self.prompt_style = 'llama'
        self.together_api_key=os.getenv("TOGETHERAI_KEY")

class Llama70BGenerator(TogetherGeneratorBase):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "meta-llama/Llama-2-70b-chat-hf"
        self.prompt_style = 'llama'
        self.together_api_key=os.getenv("TOGETHERAI_KEY")

class Gemma7BGenerator(TogetherGeneratorBase):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "google/gemma-7b-it"
        self.prompt_style = 'gemma'
        self.together_api_key=os.getenv("TOGETHERAI_KEY")

class Mixtral8x7BGenerator(TogetherGeneratorBase):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.prompt_style = 'mixtral'
        self.together_api_key=os.getenv("TOGETHERAI_KEY")

class Llama70BGeneratorNotChat(TogetherGeneratorBase):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "meta-llama/Llama-2-70b-hf"
        self.prompt_style = 'llama'
        self.together_api_key=os.getenv("TOGETHERAI_KEY")
        
# class Llama13BGenerator(OllamaGeneratorBase):
#     def __init__(self) -> None:
#         super().__init__()
#         self.model_name = "llama2:13b-chat"
#         self.prompt_style = 'llama'

# class Llama70BGenerator(OllamaGeneratorBase):
#     def __init__(self) -> None:
#         super().__init__()
#         self.model_name = "llama2:70b-chat"
#         self.prompt_style = 'llama'

# class Gemma7BGenerator(OllamaGeneratorBase):
#     def __init__(self) -> None:
#         super().__init__()
#         self.model_name = "gemma:7b-instruct"
#         self.prompt_style = 'gemma'

# class Mixtral8x7BGenerator(OllamaGeneratorBase):
#     def __init__(self) -> None:
#         super().__init__()
#         self.model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#         self.prompt_style = 'mixtral'
    