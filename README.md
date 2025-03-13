# Research Grant Toolbelt

## Overview

The Research Grant Toolbelt leverages various AI patterns - prompting, Retrieval Augmented Generation (RAG), agents, tool-calling - to help with common research grant proposal tasks. It helps with finding funding opportunities, producing application materials, and summarizing ongoing or completed projects. 

## Phase I

The first phase of this project is to develop a process to build out a budget for specific funding opportunities using a combination of the funding information, NIH budget guidelines, common assumptions, and historical data. It starts with assessing the study complexity and then proceeds on to the budget creation process, iterating through different steps to produce the final budget. Additional details are provided below.

### Task 1: Defining Your Problem and Audience

**Problem Statement:** Although the National Institutes of Health (NIH) has provided guidelines for producing required research budgets in the grant application process (https://grants.nih.gov/grants-process/write-application/advice-on-application-sections/develop-your-budget), budgets remain challenging to produce such that they adequately cover the costs to administer the research while remaining competitive as compared to other grant applications.

**Additional Context:** At the time of this midterm project, the majority of the Data Coordinating Center's funding for the research projects it supports comes from NIH grants. Ensuring the grant award amounts (which are determined using the proposed budgets) sufficiently cover research expenses is imperative to the continued financial success of the organization but also helps to streamline the center's operations in general making it possible for research personnel to focus their time on other non-administrative type activities.

### Task 2: Propose a Solution

**Solution Summary:** I propose developing a multi-agent system composed of a research or information-gathering team, a budget analysis team, and a writers team that works as follows:

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

**Tools Overview:**
* **LLM:** While I have not yet determined what model I will use for my final project, I know it would be best to use one suited to this specific domain, i.e. healthcare/research. A quick Google search suggests Med-PaLM might be a good match, but this will depend on what kind of access I can get to the model. 

    As this model does not initially appear to be supported by the orchestration framework that I intend to use - https://python.langchain.com/docs/integrations/chat/ - I will likely need to choose another. 

    I did, however finally some documentation on LangChain's website that suggests I can use a Hugging Face model instead: https://python.langchain.com/docs/integrations/llms/huggingface_endpoint/ In which case I may try to use one of the models from the Open Medical LLM Leaderboard (https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard). This will obviously require some additional testing.

    For now, I have stuck with the gpt-4o-mini model as it performs fairly well with most use cases.

* **Embedding Model:** Since this domain is especially specialized, I believe it would be best to use a fine-tuned model again specific to the healthcare domain. And to train it on a collection of funding opportunities and research project datasources identified earlier. I will also be limited in what I can do with my GPU resources in Colab as well, so I will need to stick with something comparable to the Snowflake/snowflake-arctic-embed-l used in our class assignments.

    When filtered for the MTEB (Medical, v1) benchmark, Hugging Face's Embedding Leaderboard (https://huggingface.co/spaces/mteb/leaderboard) suggests there may be several zero-shot options, e.g.

    * https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0
    * https://huggingface.co/BAAI/bge-base-en-v1.5
    * https://huggingface.co/nomic-ai/nomic-embed-text-v1-ablated

    I did some testing already with the first one and ran into some issues. I will have to continue testing to see if I can resolve those issues. For now, I have stuck with a fine-tuned Snowflake/snowflake-arctic-embed-lSnowflake/snowflake-arctic-embed-l embedding.

* **Orchestration:** I will likely stick with LangGraph for building out my agentic workflows as we have been exposed to this framework in class and it does streamline the development process. I would be interested in exploring other frameworks but think I should remain focused on accomplishing the task at hand given what I know.

    For this midterm, I set up a simple RAG pipeline using LangGraph with just 2 nodes - retrieve and generate.

* **Vector Database:** Again, I will likely stick with the vector database we have been using in our assignments - Qdrant - although I would eventually like to explore other possible solutions, e.g. FAISS. I will also need to determine whether or not it's possible for me to host this somewhere else as I will be limited most likely to what I can store in memory wherever I end up hosting this application.

    One important factor for my vector database will be maintaining metadata with the stored document embeddings such that we can recall specific pieces with specific funding opportunities and/or projects.

    I did stick with the in-memory Qdrant database for this midterm and will certainly need to change that configuration at a later time.

* **Monitoring:** Again, I will likely stick with the LangSmith framework we have used in class as this is a vey simple configuration at the beginning of the pipeline and very intuitive to understand. *NOTE: I do not have this turned on though for the quick prototype in this midterm.*

* **Evaluation:** I will also likely stick with the Ragas evaluation framework that we have used in our class assignments (and in this midterm) unless we learn of any other potentially useful solutions. 

* **User Interface:**  Initially, I didn't think a chat interface would be useful in this case, but I may be changing my mind as I find it may be useful for the end user to query the model for other grant-related information, e.g. application due dates, application specifications, application contacts, etc. It will be important to provide some example queries though to make it clear how the budget creation process will work.

    I developed a Chainlit app for this midterm project and am hosting it on Hugging Face. The public URL may be found below.

* **Agent Details:** I provided more comprehensive agent details in the Summary section above, but in summary, this solution would consist of 3 specific teams and 1 supervisor:

    * Information-Gathering Team
    * Budget Analysis Team
    * Writers Team

    I have *not* developed an agentic prototype for this midterm as I was more focused on the specific objective of demonstrating I could produce the first piece of this application, i.e. assessing the study complexity for a specific funding opportunity.

### Task 3: Dealing with the Data

**Datasources Used:**

While the exact implementation details for some of these datasources remain to be sorted, here are the datasources I think will be required:

* *NIH Funding Opportunities List:* https://grants.nih.gov/funding/nih-guide-for-grants-and-contracts (will need to be manually extracted and used to extract the funding opportunities below as there does not appear to be an API for this)

* *NIH Funding Opportunities Details:* individual URLs for funding opportunities list above

* *NIH Projects:* https://api.reporter.nih.gov/ (likely filtered for just University of Utah projects)

* *Scope Documents:* These will be strictly internal documents developed by Data Coordinating Center employees. While I do have some examples we may use, I may create test documents instead. 

* *Budget Documents:* These will also be strictly internal documents developed by Data Coordinating Center employees. I do not currently have any examples that I can use, but I can likely produce some test data that's close enough to what will be needed.

* *Effort Allocation:* This is perhaps the hardest data to account for as the data is likely incomplete and not in a format that can be easily combined with the other sources. So, again I may do some test data generation here that gets us close enough to the ideal format.

For this midterm project, I simply loaded a handful of funding opportunities. Ideally, I'd have them all pre-loaded and updated on a schedule. 

**Chunking Strategy:** The chucking strategy will depend on the datasource identified above, but I'll need to ensure the documents retain some metadata that facilitates the retrieval process at a later time. A naiive chunking strategy would not be appropriate in this case as the chat model will not be able to infer which chuck is associated with which document. 

### Task 4: Building a Quick End-to-End Pipeline

As noted above, I have *not* developed an agentic prototype for this midterm as I was more focused on the specific objective of demonstrating I could produce the first necessary piece of the budget creation process, i.e. assessing the study complexity for a specific funding opportunity.

**Public Link:** https://huggingface.co/spaces/christinemahler/AIE5-Midterm

*Note: You may also view the notebook where I developed the preliminary code and started work on some of the other components here:* https://github.com/christinemahler/AIE5/blob/main/Midterm/RAG%20Development.ipynb

### Task 5: Creating a Golden Test Data Set

*NOTE: Because I am limited in what I can do locally on my own computer, I created a "golden" test data set in the Google Colab separate from my main prototype code developed above. A link to that completed notebook is provided below.*  

**Link to Google Colab Notebook Used:** https://github.com/christinemahler/AIE5/blob/main/Midterm/Fine%20Tuning%20and%20Evaluation.ipynb

I put "golden" in quotes above because I do think my test data set needs some additional tweaking after creating it. Likely due to the size of my dataset, this process took 45 minutes to complete on each attempt and did not generate the kinds of questions I would expect for my specific use case. Rather than spend any more time on it though, I decided to proceed with the other tasks required for this midterm project. 

That being said, the initial performance is terrible:

| Metric | Score | Interpretation |
| --- | --- | --- |
| Context Recall | 0% | App is not retrieving any relevant documents |
| Faithfulness | 25% | App response is at least somewhat factually consistent with the retrieved context |
| Factual Correctness | N/A | There does not appear to be any correlation between the reference and the retrieved context (this likely makes sense given the poor quality of the test data) |
| Answer Relevancy | 10% | There is not close alignment between the user input and the response generated | 
| Context Entity Recall | 0% | There does not appear to be any correlation between the reference and the retrieved context (this likely makes sense given the poor quality of the test data) |
| Noise Sensitivity Relevant | 0% | The only good score in this evaluation, the app does not appear to be susceptible to providing incorrect responses using the provided context |

### Task 6: Fine-Tuning Open-Source Embeddings

I mentioned this above, but I did try some other embeddings instead of the Snowflake/snowflake-arctic-embed-l one we've been using in class, but I ran into either memory issues or some other issues. So, I've just stuck with using Snowflake/snowflake-arctic-embed-l to fine-tune with my funding opportunities data for the time being. A link to the embedding model is provided below:

**Public Link:** https://huggingface.co/christinemahler/aie5-midterm

The screenshot below is one of the issues encountered when trying to use a different embedding type:

![image](EmbeddingIssues.png)

### Task 7: Assessing Performance

**Fine-Tuning Evaluation Results:** While performance is improved, it's still not especially great.

| Metric | Score | Interpretation |
| --- | --- | --- |
| Context Recall | 20% | Improved a little from the base model, the app is still not doing a good job retrieving relevant documents |
| Faithfulness | 25% | Factual consistency with the retrieved context remains the same |
| Factual Correctness | 39% | There now appears to be some matching data between the reference and retrieved contexts |
| Answer Relevancy | 49% | There is closer (originally 10%) alignment between the user input and the response generated | 
| Context Entity Recall | 13% | There still is little matching data between the reference and retrieved contexts |
| Noise Sensitivity Relevant | 0% | Still the only good score in this evaluation, the app does not appear to be susceptible to providing incorrect responses using the provided context |

**Improvements:**
There is still quite a bit of work to be done to add in the other components to this application. While all of these things may not be possible to achieve in the final weeks of the course, these are the next steps I forsee needing to complete:

1) Improve Sythetically Generated Data.
2) Improve fine-tuned embedding and possibly identify alternate open-source model. Would've liked to use https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0 but was limited by GPU. Tried to use https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0 but ran into other issues that could potentially be sorted.
3) Does a multi-RAG process make sense?
4) Generate test budget data.
5) Generate test effort allocation data.
6) Determine whether data can be hosted elsewhere. 
7) Load other data sources into vector store and create retrievers.
7) Write custom tools for generating budget line items and writing documents.
8) Build agentic workflow.
9) Add monitoring.
10) Complete other relevant evaluations.
11) Decide where this will be hosted and what front-end I will use.

### Loom Video Presentation

Link to video: https://www.loom.com/share/ba91d87778b84644adf84e89723ca25c?sid=dd860fc9-4176-4911-a1ac-bbda87a65bd8