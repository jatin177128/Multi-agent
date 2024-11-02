```mermaid
    flowchart TB
    subgraph User Interface
        UI[Web Interface]
        UI --> |Company/Industry Input| C
    end

    subgraph Orchestration Layer
        C[Coordinator Agent]
        C --> R
        C --> M
        C --> RA
        C --> F
    end

    subgraph Research Layer
        R[Research Agent]
        R --> |Web Search| T1[Tavily Search]
        R --> |Industry Analysis| T2[Market Research]
        R --> |Company Profile| T3[Business Analysis]
    end

    subgraph Market Analysis Layer
        M[Market Standards Agent]
        M --> |Industry Trends| MA1[AI/ML Trends]
        M --> |Use Cases| MA2[Use Case Generator]
        M --> |Feasibility| MA3[Implementation Analysis]
    end

    subgraph Resource Layer
        RA[Resource Asset Agent]
        RA --> |Dataset Search| DS1[Kaggle]
        RA --> |Dataset Search| DS2[HuggingFace]
        RA --> |Code Search| DS3[GitHub]
    end

    subgraph Final Layer
        F[Final Proposal Agent]
        F --> |Document Generation| FP1[Proposal Creator]
        F --> |Resource Linking| FP2[Asset Linker]
    end

    subgraph Output
        FP1 --> D[Final Proposal Document]
        FP2 --> D
    end

