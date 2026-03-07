from dotenv import load_dotenv

load_dotenv()


from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import AnthropicModel
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
from deepeval.models import GPTModel
import os

model = GPTModel(model="gpt-5", base_url=os.getenv("LIGHTAI_BASE_URL"), api_key=os.getenv("LIGHTAI_API_KEY"))

answer_metric = AnswerRelevancyMetric(model=model)

test_case = LLMTestCase(
    input="Who is half blood prince?",
    actual_output="Harry Potter"

)
assert_test(test_case=test_case, metrics=[answer_metric])