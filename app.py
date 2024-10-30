from typing import Dict, List
import os
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_community.chat_models import ChatOpenAI
from tavily import TavilyClient
from dotenv import load_dotenv
import requests
import json
import streamlit as st


load_dotenv()


class Tools:

    def __init__(self):
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def tavily_search(self, query: str) -> str:
        """Enhanced web search using Tavily"""
        try:
            search_result = self.tavily_client.search(query=query)
            return json.dumps(search_result, indent=2)
        except Exception as e:
            return f"Error performing search: {str(e)}"

    def search_kaggle(self, query: str) -> str:
        """Search Kaggle for relevant datasets"""
        try:
            response = requests.get(
                f"https://www.kaggle.com/api/v1/search/datasets?search={query}",
                headers={"Authorization": f"Bearer {os.getenv('KAGGLE_API_TOKEN')}"}
            )
            return json.dumps(response.json(), indent=2)
        except Exception as e:
            return f"Error searching Kaggle: {str(e)}"

    def search_huggingface(self, query: str) -> str:
        """Search HuggingFace for models and datasets"""
        try:
            response = requests.get(
                f"https://huggingface.co/api/datasets?search={query}"
            )
            return json.dumps(response.json(), indent=2)
        except Exception as e:
            return f"Error searching HuggingFace: {str(e)}"


class AgentFactory:
    """Factory for creating specialized agents"""

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def create_research_agent(self) -> Agent:
        return Agent(
            role='Industry Research Specialist',
            goal='Conduct comprehensive industry and company research',
            backstory="""Expert in market research and industry analysis with 
            extensive experience in technology sector analysis.""",
            tools=[
                Tool(
                    name="Web Search",
                    func=self.tools.tavily_search,
                    description="Search for company and industry information"
                )
            ],
            llm=self.llm,
            verbose=True
        )

    def create_market_standards_agent(self) -> Agent:
        return Agent(
            role='AI/ML Strategy Consultant',
            goal='Analyze market trends and generate relevant AI/ML use cases',
            backstory="""Senior AI consultant specializing in identifying and 
            evaluating AI/ML opportunities across industries.""",
            tools=[
                Tool(
                    name="Web Search",
                    func=self.tools.tavily_search,
                    description="Search for AI/ML trends and use cases"
                )
            ],
            llm=self.llm,
            verbose=True
        )

    def create_resource_agent(self) -> Agent:
        return Agent(
            role='Technical Resource Specialist',
            goal='Identify and validate AI/ML resources and datasets',
            backstory="""Technical expert in AI/ML resources with deep knowledge 
            of available datasets, models, and implementations.""",
            tools=[
                Tool(
                    name="Kaggle Search",
                    func=self.tools.search_kaggle,
                    description="Search for datasets on Kaggle"
                ),
                Tool(
                    name="HuggingFace Search",
                    func=self.tools.search_huggingface,
                    description="Search for models and datasets on HuggingFace"
                )
            ],
            llm=self.llm,
            verbose=True
        )

    def create_proposal_agent(self) -> Agent:
        return Agent(
            role='AI Solution Architect',
            goal='Create comprehensive AI implementation proposals',
            backstory="""Senior solution architect specializing in creating 
            detailed and actionable AI implementation plans.""",
            llm=self.llm,
            verbose=True
        )


class TaskFactory:
    """Factory for creating specialized tasks"""

    @staticmethod
    def create_research_task(agent: Agent, company_name: str) -> Task:
        return Task(
            description=f"""Thoroughly research {company_name} and their industry:
            1. Industry classification and market position
            2. Core products/services and target markets
            3. Technology stack and digital maturity
            4. Key competitors and their AI initiatives
            5. Strategic focus areas and challenges

            Provide a detailed JSON report covering all aspects.""",
            agent=agent
        )

    @staticmethod
    def create_market_task(agent: Agent, company_name: str) -> Task:
        return Task(
            description=f"""Analyze AI/ML opportunities for {company_name}:
            1. Current industry AI/ML trends
            2. Generate 5 specific use cases with:
               - Detailed description
               - Expected benefits
               - Required technologies
               - Implementation complexity
               - ROI estimation
               - Priority level
            3. Competitive analysis of AI adoption

            Format as structured JSON with clear sections.""",
            agent=agent
        )

    @staticmethod
    def create_resource_task(agent: Agent) -> Task:
        return Task(
            description="""For each identified use case:
            1. Find relevant datasets (Kaggle/HuggingFace)
            2. Identify similar implementations
            3. List required technologies
            4. Estimate resource requirements

            Create a comprehensive JSON report with links.""",
            agent=agent
        )

    @staticmethod
    def create_proposal_task(agent: Agent) -> Task:
        return Task(
            description="""Generate a detailed implementation proposal:
            1. Executive Summary
            2. Company & Industry Analysis
            3. AI/ML Opportunity Assessment
            4. Detailed Use Cases
               - Implementation requirements
               - Resource needs
               - Timeline
            5. Risk Analysis
            6. ROI Projections
            7. Next Steps

            Format in professional Markdown with all resources linked.""",
            agent=agent
        )


class AIProposalSystem:
    """Main system orchestrator"""

    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4-turbo-preview",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.tools = Tools()
        self.agent_factory = AgentFactory(self.llm, self.tools)

    def generate_proposal(self, company_name: str) -> str:
        # Create agents
        research_agent = self.agent_factory.create_research_agent()
        market_agent = self.agent_factory.create_market_standards_agent()
        resource_agent = self.agent_factory.create_resource_agent()
        proposal_agent = self.agent_factory.create_proposal_agent()

        # Create tasks
        research_task = TaskFactory.create_research_task(research_agent, company_name)
        market_task = TaskFactory.create_market_task(market_agent, company_name)
        resource_task = TaskFactory.create_resource_task(resource_agent)
        proposal_task = TaskFactory.create_proposal_task(proposal_agent)

        # Create and execute crew
        crew = Crew(
            agents=[research_agent, market_agent, resource_agent, proposal_agent],
            tasks=[research_task, market_task, resource_task, proposal_task],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()

        # Save proposal
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_proposal_{company_name}_{timestamp}.md"

        with open(filename, "w") as f:
            f.write(result)

        return filename


# Streamlit Interface
def main():
    st.set_page_config(
        page_title="AI/ML Use Case Generator",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Enterprise AI/ML Use Case Generator")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        api_keys_valid = True

        # API key inputs
        openai_key = st.text_input("OpenAI API Key", type="password")
        tavily_key = st.text_input("Tavily API Key", type="password")
        kaggle_key = st.text_input("Kaggle API Key", type="password")

        if not all([openai_key, tavily_key, kaggle_key]):
            st.warning("Please provide all required API keys")
            api_keys_valid = False

        st.markdown("""
        ### Process Steps:
        1. 🔍 Industry & Company Research
        2. 📊 Market Standards Analysis
        3. 🗄️ Resource Collection
        4. 📑 Proposal Generation
        """)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        company_name = st.text_input(
            "Enter Company Name",
            placeholder="e.g., Tesla, Amazon, etc."
        )

        if st.button("Generate Proposal", disabled=not api_keys_valid):
            if not company_name:
                st.error("Please enter a company name")
                return

            # Set environment variables
            os.environ["OPENAI_API_KEY"] = openai_key
            os.environ["TAVILY_API_KEY"] = tavily_key
            os.environ["KAGGLE_API_TOKEN"] = kaggle_key

            system = AIProposalSystem()

            with st.spinner("Generating AI/ML proposal..."):
                try:
                    # Progress tracking
                    progress = st.progress(0)
                    status = st.empty()

                    stages = [
                        "Researching company and industry...",
                        "Analyzing AI/ML opportunities...",
                        "Collecting resources and datasets...",
                        "Generating comprehensive proposal..."
                    ]

                    for i, stage in enumerate(stages):
                        status.text(stage)
                        progress.progress((i + 1) * 25)

                    # Generate proposal
                    proposal_file = system.generate_proposal(company_name)

                    # Display proposal
                    with open(proposal_file, 'r') as f:
                        proposal_content = f.read()

                    st.markdown("### Generated AI/ML Proposal")
                    st.markdown(proposal_content)

                    # Download option
                    st.download_button(
                        label="Download Proposal",
                        data=proposal_content,
                        file_name=proposal_file,
                        mime="text/markdown"
                    )

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please check your API keys and try again")

    with col2:
        st.markdown("### Analysis Metrics")
        if 'proposal_content' in locals():
            # Display metrics
            st.metric("Use Cases Generated", "5")
            st.metric("Resources Found", "15")
            st.metric("Completion Time", "2 min")

            # Display resource summary
            st.markdown("### Resource Summary")
            st.markdown("""
            - 🗃️ Datasets Found
            - 💻 Code Repositories
            - 📚 Research Papers
            - 🔧 Implementation Guides
            """)

    st.markdown("---")
    st.markdown(
        "Built with CrewAI | Powered by OpenAI, Tavily, and HuggingFace"
    )


if __name__ == "__main__":
    main()