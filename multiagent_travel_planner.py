from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize LLM
llm = ChatOpenAI(temperature=0)

# Shared context
shared_context = {
    "location": "Goa",
    "month": "July",
    "budget": 25000,
    "transport": "road",
    "event_week": "New Year",
    "delay_hours": 6,
    "comfort_level": "medium"  # can be low, medium, high
}

# Weather Agent
weather_prompt = PromptTemplate.from_template(
    "User wants to visit {location} in {month}. Suggest indoor attractions or better months."
)
weather_chain = LLMChain(llm=llm, prompt=weather_prompt)

# Budget Agent
budget_prompt = PromptTemplate.from_template(
    "Plan a 3-day trip to {location} with a budget of ₹{budget}. Optimize transport, stay, and activities."
)
budget_chain = LLMChain(llm=llm, prompt=budget_prompt)

# Transport Agent
transport_prompt = PromptTemplate.from_template(
    "User prefers road travel to {location}. Suggest stopovers and check road conditions."
)
transport_chain = LLMChain(llm=llm, prompt=transport_prompt)

# Event Agent
event_prompt = PromptTemplate.from_template(
    "User plans to visit {location} during {event_week}. Detect peak season and recommend early booking."
)
event_chain = LLMChain(llm=llm, prompt=event_prompt)

# Itinerary Agent (collaborates with others)
itinerary_prompt = PromptTemplate.from_template(
    "Using inputs: location={location}, month={month}, budget=₹{budget}, transport={transport}, event={event_week}, delay={delay_hours} hours. Build or update a travel itinerary. Collaborate with weather, budget, transport, and event agents."
)
itinerary_chain = LLMChain(llm=llm, prompt=itinerary_prompt)

# Notification Agent
notification_prompt = PromptTemplate.from_template(
    "Notify stakeholders about a {delay_hours}-hour flight delay. Update hotel, transport, and user with revised itinerary."
)
notification_chain = LLMChain(llm=llm, prompt=notification_prompt)

# Multi-objective Optimization Agent
optimization_prompt = PromptTemplate.from_template(
    "Balance cost and comfort for a trip to {location}. Budget is ₹{budget}, comfort level is {comfort_level}. Suggest best trade-offs in transport, stay, and activities."
)
optimization_chain = LLMChain(llm=llm, prompt=optimization_prompt)

# Tools Definition
tools = [
    Tool(name="WeatherAgent", func=lambda _: weather_chain.run(shared_context), description="Handles weather-based suggestions"),
    Tool(name="BudgetAgent", func=lambda _: budget_chain.run(shared_context), description="Optimizes trip within budget"),
    Tool(name="TransportAgent", func=lambda _: transport_chain.run(shared_context), description="Suggests road travel options"),
    Tool(name="EventAgent", func=lambda _: event_chain.run(shared_context), description="Detects seasonal events and booking needs"),
    Tool(name="ItineraryAgent", func=lambda _: itinerary_chain.run(shared_context), description="Builds and updates itinerary"),
    Tool(name="NotificationAgent", func=lambda _: notification_chain.run(shared_context), description="Sends delay notifications"),
    Tool(name="OptimizationAgent", func=lambda _: optimization_chain.run(shared_context), description="Balances cost and comfort"),
]

# Orchestrator Agent
orchestrator = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

response = orchestrator.run(
    "Plan a trip to Goa in July with a ₹25,000 budget, road travel preference, during New Year week. Flight delayed by 6 hours. Balance cost and comfort."
)

print("\n🧳 Final Travel Plan:\n")
print(response)
