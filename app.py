import os
import gradio as gr
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Define State
class TicketState(TypedDict):
    ticket_text: str
    needs_fde: bool
    summary: str
    logs: list

# Simulated Jev Output Structure
class JevDecision(BaseModel):
    needs_fde: bool = Field(description="True if it's a complex API/Integration issue, False if basic account/billing issue.")
    confidence: float = Field(description="Confidence score between 0 and 1.")

# Initialize Models
# We use Gemini to simulate both Jev (System 1) and the heavy LLM (System 2)
# In production, Jev would be a fast, cheap API call to TypeSafe AI.
try:
    # Use a small fast model for Jev simulation
    jev_sim_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    # Use a larger model for the actual summarization
    heavy_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)
except Exception as e:
    print(f"Error initializing models: {e}. Please ensure GOOGLE_API_KEY is set.")
    
# Node 1: Jev Classifier
def jev_classifier_node(state: TicketState):
    ticket = state["ticket_text"]
    state["logs"].append(f"JEV CLASSIFIER: Analyzing ticket...")
    
    prompt = f"Ticket: {ticket}\n\nIs this a complex technical API/Integration issue (True) or a basic account/billing issue (False)?"
    
    # Force structured output to simulate Jev's decision capabilities
    structured_llm = jev_sim_llm.with_structured_output(JevDecision)
    decision = structured_llm.invoke(prompt)
    
    state["needs_fde"] = decision.needs_fde
    state["logs"].append(f"JEV DECISION: FDE Required = {decision.needs_fde} (Confidence: {decision.confidence})")
    
    return state

# Node 2: Tier 1 Support (Basic Issue)
def tier_1_node(state: TicketState):
    state["logs"].append("ROUTING: Routing to Tier 1 Support.")
    state["summary"] = "Auto-closed: Issue sent to Tier 1 support. FDE intervention not required."
    return state

# Node 3: LLM Summarizer (Complex Issue)
def llm_summarizer_node(state: TicketState):
    ticket = state["ticket_text"]
    state["logs"].append("ROUTING: Routing to FDE. Starting LLM Summarizer...")
    
    prompt = f"Ticket: {ticket}\n\nExtract the API endpoint that is failing, the error code, and summarize the issue in exactly 3 bullet points for a senior engineer."
    
    response = heavy_llm.invoke(prompt)
    state["summary"] = response.content
    state["logs"].append("LLM SUMMARIZER: Summary generated.")
    return state

# Routing Logic
def route_ticket(state: TicketState):
    if state["needs_fde"]:
        return "llm_summarizer"
    return "tier_1_support"

# Build LangGraph
workflow = StateGraph(TicketState)

workflow.add_node("jev_classifier", jev_classifier_node)
workflow.add_node("tier_1_support", tier_1_node)
workflow.add_node("llm_summarizer", llm_summarizer_node)

workflow.set_entry_point("jev_classifier")
workflow.add_conditional_edges(
    "jev_classifier",
    route_ticket,
    {
        "tier_1_support": "tier_1_support",
        "llm_summarizer": "llm_summarizer"
    }
)
workflow.add_edge("tier_1_support", END)
workflow.add_edge("llm_summarizer", END)

app_graph = workflow.compile()

# Gradio Interface Function
def process_ticket(ticket_input):
    if not os.environ.get("GOOGLE_API_KEY"):
        return "Error: GOOGLE_API_KEY environment variable not set.", ""
        
    initial_state = {"ticket_text": ticket_input, "needs_fde": False, "summary": "", "logs": []}
    final_state = app_graph.invoke(initial_state)
    
    log_output = "\n".join(final_state["logs"])
    return final_state["summary"], log_output

# Gradio UI
with gr.Blocks(title="FDE Ticket Escalator") as demo:
    gr.Markdown("# FDE Ticket Escalator (Powered by Jev & LangGraph)")
    gr.Markdown("Paste a customer support ticket below. The **Jev Classifier** will decide if it needs an FDE. If it does, the **LLM** will summarize it.")
    
    with gr.Row():
        with gr.Column():
            ticket_input = gr.Textbox(lines=5, placeholder="Enter customer ticket here...", label="Incoming Customer Ticket")
            submit_btn = gr.Button("Process Ticket")
            
            gr.Examples(
                examples=[
                    "Hi, I forgot my admin password, please help.",
                    "Hey, our POST request to /v2/users is failing with a 500 timeout error when the payload exceeds 2MB."
                ],
                inputs=ticket_input
            )
            
        with gr.Column():
            output_summary = gr.Textbox(lines=5, label="Final Output / Summary")
            output_logs = gr.Textbox(lines=5, label="System Logs (Jev Routing)")

    submit_btn.click(fn=process_ticket, inputs=ticket_input, outputs=[output_summary, output_logs])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
