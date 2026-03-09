from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv("apikey.env")
class Agent:

    def __init__(self, medical_report=None, role=None, extra_info=None):
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info

        self.prompt_template = self.create_prompt_template()

        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def create_prompt_template(self):

        if self.role == "MultidisciplinaryTeam":

            template = f"""
Act as a multidisciplinary medical team.

You receive reports from specialists.

Cardiologist Report:
{self.extra_info.get("cardiologist_report")}

Psychologist Report:
{self.extra_info.get("psychologist_report")}

Pulmonologist Report:
{self.extra_info.get("pulmonologist_report")}

Task:
Identify the top 3 possible health issues and explain why.
Return bullet points.
"""

        else:

            templates = {

                "Cardiologist": """
You are a cardiologist.

Analyze the medical report and identify possible cardiac issues.

Medical Report:
{medical_report}
""",

                "Psychologist": """
You are a psychologist.

Analyze the patient's mental health condition.

Medical Report:
{medical_report}
""",

                "Pulmonologist": """
You are a pulmonologist.

Analyze the respiratory symptoms.

Medical Report:
{medical_report}
"""
            }

            template = templates[self.role]

        return PromptTemplate.from_template(template)

    def run(self):

        prompt = self.prompt_template.format(
            medical_report=self.medical_report
        )

        response = self.model.invoke(prompt)

        return response.content


class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Cardiologist")


class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Psychologist")


class Pulmonologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Pulmonologist")


class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_report, pulmonologist_report):

        extra = {
            "cardiologist_report": cardiologist_report,
            "psychologist_report": psychologist_report,
            "pulmonologist_report": pulmonologist_report
        }

        super().__init__(role="MultidisciplinaryTeam", extra_info=extra)