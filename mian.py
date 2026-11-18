import os
import csv
import json
from threading import Thread
from kivy.app import App
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.properties import StringProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.behaviors import ButtonBehavior
from kivy.core.window import Window 
from kivy.properties import ListProperty
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool  # Assuming tools.py is in the same directory
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()

# Simple user storage
USERS_FILE = "users.json"
CHROMA_PATH = "chroma_db"







def load_users():
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}





def save_users(users: dict):
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
    except Exception:
        pass





def save_symptoms_entry(symptoms: str, disease: str, json_file: str = "symptoms_disease.json"):
    entries = []
    try:
        # If file exists but is empty or invalid, treat as empty list
        if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
            with open(json_file, "r", encoding="utf-8") as jf:
                entries = json.load(jf)
                if not isinstance(entries, list):
                    entries = []
    except Exception as e:
        print("Warning: couldn't read existing symptoms file:", e)
        entries = []

    entries.append({"symptoms": symptoms, "disease": disease})

    # Write atomically to avoid partial writes
    tmp_file = json_file + ".tmp"
    try:
        with open(tmp_file, "w", encoding="utf-8") as jf:
            json.dump(entries, jf, indent=2)
            jf.flush()
            os.fsync(jf.fileno())
        os.replace(tmp_file, json_file)
        print(f"Saved symptoms entry to {json_file}")
        return True
    except Exception as e:
        print("Error saving symptoms entry:", e)
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass
        return False






def save_treatment_entry(disease: str, treatment: str, json_file: str = "disease_treatment.json"):
    entries = []
    try:
        if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
            with open(json_file, "r", encoding="utf-8") as jf:
                entries = json.load(jf)
                if not isinstance(entries, list):
                    entries = []
    except Exception as e:
        print("Warning: couldn't read existing treatment file:", e)
        entries = []

    entries.append({"disease": disease, "treatment": treatment})

    tmp_file = json_file + ".tmp"
    try:
        with open(tmp_file, "w", encoding="utf-8") as jf:
            json.dump(entries, jf, indent=2)
            jf.flush()
            os.fsync(jf.fileno())
        os.replace(tmp_file, json_file)
        print(f"Saved treatment entry to {json_file}")
        return True
    except Exception as e:
        print("Error saving treatment entry:", e)
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass
        return False
    




def save_healthcare_entry(age: str, gender: str, weight: str, height: str, sickness: str, tip: str, json_file: str = "healthcare_tip.json"):
    entries = []
    try:
        if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
            with open(json_file, "r", encoding="utf-8") as jf:
                entries = json.load(jf)
                if not isinstance(entries, list):
                    entries = []
    except Exception as e:
        print("Warning: couldn't read existing healthcare file:", e)
        entries = []

    entries.append({"age": age, "gender": gender, "weight": weight, "height": height, "sickness": sickness, "healthcare_tip": tip})

    tmp_file = json_file + ".tmp"
    try:
        with open(tmp_file, "w", encoding="utf-8") as jf:
            json.dump(entries, jf, indent=2)
            jf.flush()
            os.fsync(jf.fileno())
        os.replace(tmp_file, json_file)
        print(f"Saved healthcare entry to {json_file}")
        return True
    except Exception as e:
        print("Error saving healthcare entry:", e)
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass
        return False
    




class ResearchResponse(BaseModel):
    topic: str
    text: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert healthcare agent.
            Answer the user query, use necessary tools
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())
tools = [search_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm, 
    prompt=prompt,
    tools=tools

)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


class LoadingPopup(Popup):
    def __init__(self, **kwargs):
        super().__init__(
            title="Generating AI response",
            content=Label(
                text="Please wait...",
                halign="center",
                valign="middle",
                font_size="22dp",
            ),
            size_hint=(0.55, 0.25),
            auto_dismiss=False,
            **kwargs,
        )


def run_ai_request(query: str, on_complete):
    loading_popup = LoadingPopup()
    loading_popup.open()

    def worker():
        try:
            raw_response = agent_executor.invoke({"query": query})
            structured_response = parser.parse(raw_response.get("output"))
            result = (structured_response, None)
        except Exception as error:
            print("Error generating AI response:", error)
            result = (None, error)

        def finish(_dt):
            loading_popup.dismiss()
            on_complete(*result)

        Clock.schedule_once(finish)

    thread = Thread(target=worker, daemon=True)
    thread.start()
'''
go_again = True

while go_again:
    raw_response = agent_executor.invoke({"query": f"What disease does the patient have if they have {self.symptoms} as there symptoms?"})
    #print(raw_response)
    try:
        structured_response = parser.parse(raw_response.get("output"))
        print("Topic:", structured_response.topic)
        print("\n")
        print(structured_response.text)
        print("\n")
        print("In conclusion:", structured_response.summary)
        print("\n")
        print("The sources I used:", structured_response.sources)
    except Exception as e:
        print("Error parsing response:", e, "Raw response:", raw_response)

    go_again = input("Do you have another question? (y/n): ").lower()

    if go_again == 'y':
        go_again = True
    else:
        go_again = False
        print("Goodbye! Thank you for using the research agent!")
'''
Window.clearcolor = 0, 1, 0.75, 0,





class LoginScreen(Screen):
    def login_user(self, role: str):
        username = self.ids.login_username.text.strip()
        password = self.ids.login_password.text.strip()

        if not username or not password:
            if hasattr(self.ids, "login_message"):
                self.ids.login_message.text = "Enter username and password"
            return

        users = load_users()
        user = users.get(username)
        if user and user.get("password") == password and user.get("role") == role:
            App.get_running_app().current_user = username
            App.get_running_app().root.current = role
            if hasattr(self.ids, "login_message"):
                self.ids.login_message.text = "Login successful"
        else:
            if hasattr(self.ids, "login_message"):
                self.ids.login_message.text = "Invalid username/password or role"





class RegisterScreen(Screen):
    def register_user(self, role: str):
        first = self.ids.first_name_input.text.strip()
        last = self.ids.last_name_input.text.strip()
        username = self.ids.username_input.text.strip()
        password = self.ids.password_input.text.strip()
        age = self.ids.age_input.text.strip()
        gender = self.ids.gender_input.text.strip()
        height = self.ids.height_input.text.strip()
        weight = self.ids.weight_input.text.strip()
        sickness = self.ids.sickness_input.text.strip()

        # Check if all fields are filled
        if not all([first, last, username, password, age, gender, height, weight, sickness]):
            if hasattr(self.ids, "register_message"):
                self.ids.register_message.text = "Please fill in all fields"
            return False

        users = load_users()
        if username in users:
            if hasattr(self.ids, "register_message"):
                self.ids.register_message.text = "Username already exists"
            return False

        users[username] = {"first": first, "last": last, "password": password, "role": role, "age": age, "gender": gender, "height": height, "weight": weight, "sickness": sickness}
        save_users(users)
        if hasattr(self.ids, "register_message"):
            self.ids.register_message.text = "Registration successful!"
        
        # Store current user and navigate to the appropriate screen based on role
        app = App.get_running_app()
        app.current_user = username
        screen_name = "doctor" if role == "doctor" else "patient"
        app.root.current = screen_name
        return True





class DoctorScreen(Screen):
    def on_enter(self, *args):
        app = App.get_running_app()
        username = getattr(app, "current_user", "")
        welcome = "Welcome Doctor"
        if username:
            users = load_users()
            user = users.get(username)
            if user:
                first = user.get("first", "")
                last = user.get("last", "")
                full = (first + " " + last).strip()
                if full:
                    welcome = f"Welcome {full}"

        try:
            if hasattr(self.ids, "doctor_welcome"):
                self.ids.doctor_welcome.text = welcome
        except Exception:
            pass





class PatientScreen(Screen):
    def on_enter(self, *args):
        app = App.get_running_app()
        username = getattr(app, "current_user", "")
        welcome = "Welcome Patient"
        if username:
            users = load_users()
            user = users.get(username)
            if user:
                first = user.get("first", "")
                last = user.get("last", "")
                full = (first + " " + last).strip()
                if full:
                    welcome = f"Welcome {full}"

        try:
            if hasattr(self.ids, "patient_welcome"):
                self.ids.patient_welcome.text = welcome
        except Exception:
            pass


    healthcare_tip = StringProperty("")
    

   # patient_age = StringProperty("")
   # patient_gender = StringProperty("")
   # patient_weight = StringProperty("")
   # patient_height = StringProperty("")
   # patient_sickness = StringProperty("")
    def load_documents():
        DATA_PATH = os.path.join(os.path.dirname(__file__), "Data")
        document_loader = PyPDFDirectoryLoader(DATA_PATH)
        return document_loader.load()




    def split_documents(documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)




    def get_embedding_function():
        embeddings = BedrockEmbeddings(
            credentials_profile_name="default", region_name="us-east-1"
        )
        return embeddings



    '''
    def add_to_chroma(chunks: list[Document]):
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function= self.get_embedding_function()
        )
        # Generate IDs if needed
        chunk_ids = [f"doc_{i}" for i in range(len(chunks))]
        new_chunks = []
        for chunk in chunk_ids:
            new_chunks.append(chunk)
        
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
        return db
    '''
    def on_press_submit_healthcare_tips(self):
        app = App.get_running_app()
        username = getattr(app, "current_user", "")

        if username:
            users = load_users()
            self.patient_age = users.get(username, {}).get('age')
            self.patient_gender = users.get(username, {}).get('gender')
            self.patient_weight = users.get(username, {}).get('weight')
            self.patient_height = users.get(username, {}).get('height')
            self.patient_sickness = users.get(username, {}).get('sickness')
            query = f"How much should a {self.patient_gender} person who is {self.patient_age} year old weighing {self.patient_weight} lbs and has a height of {self.patient_height} feet with a sickness of {self.patient_sickness} exercise daily and how much should there calorie intake be?"

            def on_complete(structured_response, error):
                if error:
                    self.healthcare_tip = "Unable to generate healthcare tips. Please try again."
                    return
                self.healthcare_tip = f"If you are a {self.patient_gender} person who is {self.patient_age} years old, weighing {self.patient_weight} lbs and your height is {self.patient_height} feet with a sickness of {self.patient_sickness}, here is your healthcare tip:\n\n{structured_response.text}\n\nSources used:\n{structured_response.sources}"
                save_healthcare_entry(
                    self.patient_age,
                    self.patient_gender,
                    self.patient_weight,
                    self.patient_height,
                    self.patient_sickness,
                    self.healthcare_tip,
                )

            run_ai_request(query, on_complete)





class HealthTipScreen(Screen):
    new_healthcare_tip = StringProperty("")

    patient_age = StringProperty("")
    patient_gender = StringProperty("")
    patient_weight = StringProperty("")
    patient_height = StringProperty("")
    patient_sickness = StringProperty("")
    def on_press_generate_healthcare_tip(self):
        self.patient_age = self.ids.age_input_h.text
        self.patient_gender = self.ids.gender_input_h.text
        self.patient_weight = self.ids.weight_input_h.text
        self.patient_height = self.ids.height_input_h.text
        self.patient_sickness = self.ids.sickness_input_h.text
        query = f"Give me a list of supplements, over the counter treatment suggestions, and cures for the symptoms of {self.patient_sickness} for a {self.patient_gender} person who is {self.patient_age} years old weighing {self.patient_weight} lbs and has a height of {self.patient_height} feet. Also, how much should they exercise daily and how much should there calorie intake be?"

        def on_complete(structured_response, error):
            if error:
                self.new_healthcare_tip = "Unable to generate healthcare tips. Please try again."
                return
            self.new_healthcare_tip = f"If you are a {self.patient_gender} person who is {self.patient_age} years old, weighing {self.patient_weight} lbs and your height is {self.patient_height} feet with a sickness of {self.patient_sickness}, here is your healthcare tip:\n\n{structured_response.text}\n\nSources used:\n{structured_response.sources}"
            save_healthcare_entry(
                self.patient_age,
                self.patient_gender,
                self.patient_weight,
                self.patient_height,
                self.patient_sickness,
                self.new_healthcare_tip,
            )

        run_ai_request(query, on_complete)





class DiseaseScreen(Screen):
    symptoms = StringProperty("")
    disease = StringProperty("")
    def on_press_button_disease(self):
        self.symptoms = self.ids.symptoms_input.text

        def on_complete(structured_response, error):
            if error:
                self.disease = "Unable to analyze the symptoms. Please try again."
                return
            self.disease = f"The possible diseases based on {self.symptoms} the patient may have are:\n\n{structured_response.summary}\n\nSources used:\n{structured_response.sources}"
            save_symptoms_entry(self.symptoms, self.disease)

        run_ai_request(
            f"What disease does the patient have if they have {self.symptoms} as there symptoms?",
            on_complete,
        )




class ChatScreen(Screen):
    chat_response = StringProperty("")
    chat_input = StringProperty("")
    def on_press_button_chat(self):
        user_query = self.ids.chat_input.text

        def on_complete(structured_response, error):
            if error:
                self.chat_response = "Unable to generate a response. Please try again."
                return
            self.chat_response = f"Response to your query:\n\n{structured_response.text}\n\nSources used:\n{structured_response.sources}"

        run_ai_request(user_query, on_complete)





class ChatDScreen(Screen):
    chatd_response = StringProperty("")
    chatd_input = StringProperty("")
    def on_press_button_chatd(self):
        user_query = self.ids.chatd_input.text

        def on_complete(structured_response, error):
            if error:
                self.chatd_response = "Unable to generate a response. Please try again."
                return
            self.chatd_response = f"Response to your query:\n\n{structured_response.text}\n\nSources used:\n{structured_response.sources}"

        run_ai_request(user_query, on_complete)




class TreatmentScreen(Screen):
    treatment_disease = StringProperty("")
    treatment_cure = StringProperty("")
    def on_press_button_treatment(self):
        self.treatment_disease = self.ids.treatment_disease_input.text

        def on_complete(structured_response, error):
            if error:
                self.treatment_cure = "Unable to find a treatment response. Please try again."
                return
            self.treatment_cure = f"The treatment for {self.treatment_disease} is:\n\n{structured_response.summary}\n\nSources used:\n{structured_response.sources}"
            save_treatment_entry(self.treatment_disease, self.treatment_cure)

        run_ai_request(
            f"What is the treatment for {self.treatment_disease}?",
            on_complete,
        )




class WindowManager(ScreenManager):
    pass



class RoundedButton(ButtonBehavior, Label):
    background_color = ListProperty([1, 1, 1, 1])



class HealthcareApp(App):
      def build(self):
        return kv
      



kv = Builder.load_file("HealthcareApp.kv")



if __name__ == '__main__':
    HealthcareApp().run()
