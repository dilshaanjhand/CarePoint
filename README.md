# CarePoint

CarePoint is a healthcare assistant desktop app built with Python, Kivy, and LangChain. Patients and doctors can register, log in, and use an AI agent (GPT-4o-mini). Doctors can use it to look up possible diseases from symptoms, find treatments, and chat with a healthcare assistant. Patients can get personalized health tips and chat with a healthcare assistant.

> **Disclaimer:** CarePoint is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

## Features

CarePoint has two account types. Each one gets its own home screen.

### Patients
- **Personalized healthcare tips:** one click gives exercise and calorie advice based on the age, gender, weight, height, and condition entered at registration
- **Additional healthcare:** enter any age, gender, weight, height, and symptoms to get supplement suggestions, over-the-counter treatments, and daily exercise and calorie recommendations
- **Chat with the expert agent:** ask the AI healthcare assistant anything

### Doctors
- **Disease detection:** enter a patient's symptoms (separated by commas) and get possible diseases
- **Treatment lookup:** enter a disease and get treatment information
- **Chat with the expert agent:** a separate chat for medical questions

### For everyone
- Register as a patient or a doctor, then log in with a username and password
- Answers include the sources the agent used
- Results are saved to local JSON files

## Requirements

- Python 3.12
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Installation

```bash
git clone https://github.com/dilshaanjhand/CarePoint.git
cd CarePoint
pip install -r requirements.txt
```

Create a file named `.env` in the project folder:

```
OPENAI_API_KEY=your_key_here
```

## Usage

```bash
python mian.py
```

1. On the login screen, click **Register Here**.
2. Fill in your details and click **Sign Up as Patient** or **Sign Up as Doctor**. Both account types fill in the same registration form, so doctors also enter age, gender, weight, height, and symptoms.
3. Use the buttons on your home screen to pick a feature.
4. Click **Sign Out** to return to the login screen.

When you log in later, use the button that matches your account type (**Login as Patient** or **Login as Doctor**).

## Project Structure

| File | Purpose |
|------|---------|
| `mian.py` | Main app: Kivy screens, login and registration, and the AI agent |
| `HealthcareApp.kv` | Kivy layout for the user interface |
| `tools.py` | Tools the agent can use (web search and saving output) |
| `retrieval_agent.py` | Standalone command-line research agent (not used by the GUI) |
| `symptoms_disease.json`, `disease_treatment.json`, `healthcare_tip.json` | Saved results; created and updated while the app runs |
| `users.json` | Created on first registration (not included in the repo) |
| `requirements.txt` | Python dependencies |

## Limitations

- AI-generated answers can be wrong and are not a medical diagnosis.
- Passwords are stored in plain text in a local `users.json` file. This is a student project, so do not reuse a real password.
- Requires an internet connection and an OpenAI API key.

## License

MIT. See the [LICENSE](LICENSE) file.

## Author

Made by [dilshaanjhand](https://github.com/dilshaanjhand).
