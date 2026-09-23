from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class Diagnosis(BaseModel):
    severity: str
    suspected_cause: str
    reason: str
    confidence: float
    next_action: str

response = client.responses.parse(
    model="gpt-5.6-luna",
    input=[{"role": "system", "content": ("You are a professional AI Cluster Engineer at diagnosing performance issues. "
                                         "You will be given a description of a performance issue and you will provide a diagnosis with given structure."),},
           {"role": "user", "content": """inter-node NCCL allreduce test recorded 75% of the expected throughput, the test run on 4 nodes with 8 GPUs each. The network is InfiniBand."""},
    ],
    text_format=Diagnosis
)

event = response.output_parsed
print(event)
print(f"Severity: {event.severity}")
print(f"Suspected Cause: {event.suspected_cause}")
print(f"Confidence: {event.confidence}")
print(f"Next Action: {event.next_action}")
