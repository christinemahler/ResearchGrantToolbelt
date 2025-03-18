# Research Grant Toolbelt
*Note: The following graph is just a preliminary representation of the tool while it's still actively in development.*

![image](Images/Research%20Grant%20Toolkit%20NEW.png)

## Overview

The Research Grant Toolbelt leverages various AI patterns - prompting, Retrieval Augmented Generation (RAG), agents, tool-calling - to help with common research grant proposal tasks. It helps with finding funding opportunities, producing application materials, and summarizing ongoing or completed projects. 

[Link to Presentation Slides](https://www.canva.com/design/DAGhoBwj0NI/0EAwY1mNZzq4WqlTZ7U9HQ/view?utm_content=DAGhoBwj0NI&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h96b31b7c49)

## Phase I

The first phase of this project is to develop a process to build out a budget for specific funding opportunities using a combination of the funding information, NIH budget guidelines, common assumptions, and historical data. It starts with assessing the study complexity and then proceeds on to the budget creation process, iterating through different steps to produce the final budget. Additional details are provided below.

### Task 1: Defining Your Problem and Audience

**Problem Statement:** Although the National Institutes of Health (NIH) has provided guidelines for producing required research budgets in the grant application process (https://grants.nih.gov/grants-process/write-application/advice-on-application-sections/develop-your-budget), budgets remain challenging to produce such that they adequately cover the costs to administer the research while remaining competitive as compared to other grant applications.

**Additional Context:** At the time of this midterm project, the majority of the Data Coordinating Center's funding for the research projects it supports comes from NIH grants. Ensuring the grant award amounts (which are determined using the proposed budgets) sufficiently cover research expenses is imperative to the continued financial success of the organization but also helps to streamline the center's operations in general making it possible for research personnel to focus their time on other non-administrative type activities.

### Task 2: Propose a Solution

> **Note:** The solution changed during the development process due to issues encountered during the testing process. Both the initial solution and the final solution are provided below.

**Initial Solution Summary:** I propose developing a multi-agent system composed of a research or information-gathering team, a budget analysis team, and a writers team that works as follows:

* *Information-Gathering Team:* Collects the information needed to produce a comprehensive budget. This could include all of the following:
    * Funding Opportunity Information (https://grants.nih.gov/funding/nih-guide-for-grants-and-contracts)
    * Funded Research Projects (https://reporter.nih.gov/)
    * Historical Budgets (proprietary)
    * Effort Allocation information (proprietary)
    * Scope Assessments (proprietary)
* *Budget Analysis Team:* Uses the collected information to calculate budget totals. 
* *Writers Team:* Combines all the details from above to produce a budget in additional to several other possible documents:
    * Recommended Questions for Follow-Up
    * Study Complexity Assessment
    * Project Summary
    * Budget Justification

The budget creation process will necessarily be a multi-step process beginning with a study complexity assessment and ending with a preliminary budget that may be used in the budget creation process. Additional enhancements to the system may allow an end user to upload a budget or recall a budget from memory and make tweaks to it as needed.

**Final Solution Summary:** Upon developing and testing the initial proposed solution above, it was found that the application did not perform quite as well as intended. Going from one team to another would often result in important information getting lost along the way. The solution was to move the budget analysis and documentation components within the research team itself and to develop more controlled, asynchronous pipelines for completing the different parts of the budget documentation. The final architecture is now as follows:

- *Opportunities Information Retriever Agent:* Provides general information about specific funding opportunities using a naive RAG pattern.
- *Budget Generator Agent:* Assesses study complexity for provided funding opportunity and passes to the following agent flows for further processing:
    - *Personnel Effort Agent:* Estimates personnel effort using the provided study complexity.
    - *Personnel Effort Writer Agent:* Writes the estimated personnel effort to a text file.
    - *Personnel Costs Agent:* Calculates personnel costs from the estimated effort allocations.
    - *Personnel Costs Writer Agent:* Writes the personnel costs to a text file.
    - *Personnel Justifications Agent:* Uses the estimated personnel effort to produce a list of justifications.
    - *Personnel Justifications Writer Agent:* Writes the estimated personnel justifications to a text file.
    - *Non-Personnel Costs Agent:* Estimates non-personnel costs using the provided study complexity.
    - *Non-Personnel Costs Writer Agent:* Writes the estimated non-personnel costs to a text file.
    - *Non-Personnel Costs Justification Agent:* Uses the estimated non-personnel costs to produce a list of justifications.
    - *Non-Personnel Costs Justification Writer Agent:* Writes the estimated non-personnel costs justifications to a text file.
    - *Final Budget Writer Agent:* Combines all the information from the budget workflows above to produce a final budget document.
    - *Dopifier Agent:* Generates a downloadable PDF version of the final budget document.
- *Project Information Retriever Agent:* Provides general information about specific NIH projects using a naive RAG pattern.
- *Opportunities Summary Agent:* Provides summary-level funding opportunity information using a naive RAG pattern.
- *Projects Summary Agent:* Provides summary-level projects information using a naive RAG pattern.
- *Publication Search Agent:* Searches PubMed site for relevant publications.
- *Search Agent:* Searches internet for relevant information.

**Tools Overview:**

> **Note:** Several of these tools will be modified in the production phase of this project as the need for data protections will arise when integrating effort allocation and historical budget data.

* **LLM Models:** For the first phase of this project, I have stuck with an assortment of OpenAI models as follows:

    - small_llm = gpt-4o-mini
    - medium_llm = gpt-4o
    - large_llm = gpt-4-turbo
    - xlarge_llm = o1

    Different models are used for varying levels of complexity with the supervisor and budget analysis agents using the large_llm and the others using a smaller model. The xlarge_llm model is included only for testing purposes at this time.

    For future phases and given the sensitivity of the effort allocation and budget data when fully integrated, it will be essential to use a self-hosted LLM for at least the budget generator process and for which we can likely stand up a private Hugging Face inference endpoint: https://huggingface.co/inference-endpoints/dedicated

* **Embedding Model:** 

* **Orchestration:** LangGraph

* **Vector Database:** Three collections - Opportunities, Projects, and Opportunities Sumamry - are stored in a cloud-hosted Qdrant database. The data is loaded initially through a separate process and ideally would be maintained using some automated pipeline in the future. 

    Additional work was started on an in-memory database that loads budget guidelines from NIH's website. These guidelines would be used to answer general questions about the budget generation process once again following a naive RAG pattern.

* **Monitoring:** LangSmith (turned off for the time being)

* **Evaluation:** 

* **User Interface:**  Chainlit

* **Agent Tools:** 
    - *assess_study_complexity:* Uses a naive RAG pattern and prompt to assess the study complexity for the requested funding opportunity.
    - *retrieve_opportunities_information:* Uses a naive RAG pattern to answer general questions about funding opportunities.
    - *retrieve_projects_information:* Uses a naive RAG pattern to answer general questions about NIH projects.
    - *summarize_opportunities:* Uses a naive RAG pattern to summarize information about funding opportunities.
    - *summarize_projects:* Uses a naive RAG pattern to summarize information about NIH projects.
    - *calculate_person_months:* Calculates person months using effort allocation %.
    - *read_document:* Reads documents.
    - *write_document:* Writes documents.
    - *edit_document:* Edits documents.
    - *combine_text_files:* Combines documents into a single text file. The final combined file is explicitly named for the time being to control for issues with downloading. THis may be revised at a later time.
    - *save_text_as_markdown:* Converts a text file to markdown to prepare for converting to PDF. 
    - *convert_markdown_to_pdf_using_pdfkit:* Converts a markdown file to PDF.
    - *convert_txt_to_docx:* Not currently used, this tool converts a text file to a .docx file.
    - *remove_text_files:* Removes all text files within the specified directory.

### Task 3: Dealing with the Data

> **Note:** Some of the datasources - Scope Documents, Budget Documents, Effort Allocation - have not yet been integrated into this application as they are still in the process of being finalized and/or collected. They will also need to be protected.

**Datasources:**

* *NIH Funding Opportunities List:* https://grants.nih.gov/funding/nih-guide-for-grants-and-contracts (will need to be manually extracted and used to extract the funding opportunities below as there does not appear to be an API for this)

* *NIH Funding Opportunities Details:* individual URLs for funding opportunities list above

* *NIH Projects:* https://api.reporter.nih.gov/ (likely filtered for just University of Utah projects)

* *Scope Documents:* These will be strictly internal documents developed by Data Coordinating Center employees. They will provide some additional scoping information relevant to the budget generation process.

* *Budget Documents:* These will also be strictly internal documents developed by Data Coordinating Center employees. They include historical budget documents that may be used to fine-tune the budget generation process.

* *Effort Allocation:* Strictly internal actual effort allocation data recorded by Data Coordinating Center employees.

**Chunking Strategy:** The chucking strategy depends on the datasource identified above but requires the documents retain some metadata that facilitates the retrieval process at a later time. 

### Task 4: Building a Quick End-to-End Pipeline

- **Application:** 
- **Application Development Notebook:**
- **Data Load Notebook:** 

### Future Phases

Future phases of this project will mostly involve making this application production-ready and adding additional features. 

**Phase 2 (Budget Generation Fine-Tuning):**
- Add historical budgets, effort allocation, and scope documents to vector store
- Develop additional agents for budget fine-tuning
- Change LLM models for budget generation workflow

**Phase 3 (Production Readiness):**
- Finalize vector store hosting location
- Finalize application hosting location
- Build/schedule data pipelines for all data sources
- Additional fine-tuning and performance optimization as needed
- Establish cost control measures
- Determine deployment model (CI/CD)
- Establish monitoring and maintenance plan
- Create security plan

**Phase 4 (Production):**
- Deployment
- Distribution

**Phase 5 (Enhancements):**
- Add additional features, e.g. support for developing other grant application materials

**Phase 6 (Growth):**
- Add support for other non-NIH funding opportunities