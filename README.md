# Cybersecurity-Graph-RAG  
This project aims to advance cybersecurity knowledge dissemination, visualization, and analysis of threat actor relationships in China. It introduces multiple innovations beyond existing research and tools.  
The knowledge graph platform uniquely integrates three core cybersecurity domains: software vulnerabilities, attack group alerts, and APT (Advanced Persistent Threat) organization profiles—ensuring comprehensive coverage. It supports both direct and multi-hop searches, enhanced by real-time query suggestions, fuzzy matching, and personalized recommendations to deepen user exploration. Built with a modular architecture—decoupling front-end, back-end, and database—the system is designed for maintainability and future scalability. Additional features include interactive statistical visualizations and keyword-based tracking of APT activities.  
A key technical innovation lies in the search algorithm. While leveraging state-of-the-art Retrieval-Augmented Generation (RAG), the project tailors the pipeline specifically for Neo4j graph databases. This adaptation enables large language models to effectively retrieve and reason over graph-structured data. The optimized workflow dramatically reduces token usage and improves search efficiency, while preserving both breadth (coverage of relevant nodes) and depth (multi-hop reasoning) in responses. As a result, the system delivers accurate, context-aware answers with strong inferential capabilities—bridging the gap between unstructured natural language queries and structured cyber threat intelligence.  
Overall, the platform not only consolidates fragmented cybersecurity knowledge into a unified, searchable graph but also pioneers an efficient, graph-native RAG framework that sets a new standard for domain-specific QA systems in threat intelligence.  
Innovative Knowledge Graph-based RAG Algorithm Pipeline:  
<img width="1735" height="1034" alt="flow" src="https://github.com/user-attachments/assets/d6b71d40-ba47-4251-883a-664f34cc9d67" />  

## Framework  
<img width="845" height="486" alt="image" src="https://github.com/user-attachments/assets/a9141af7-acd8-4f17-a0ce-2dd704456d8d" />  
<img width="944" height="266" alt="image" src="https://github.com/user-attachments/assets/09345d25-ef1e-426a-8ce1-6b02852344e6" />  
  
## Module Functionality Description  
The platform is built on a Neo4j database containing three distinct, domain-separated graph datasets: attack patterns and workflows, cybersecurity vulnerabilities, and global APT (Advanced Persistent Threat) organizations. All platform features are implemented based on this integrated data foundation.   

### Real-time News Module    
This module uses web crawlers to continuously fetch the latest cybersecurity-related news and forum posts from Baidu search results, displaying headlines with clickable links so users can stay updated on emerging threats and industry trends.  
### Graph Visualization Module  
Leveraging the ECharts library and connected to the Neo4j database, this module visualizes subgraphs upon user query—for example, entering a vulnerability name or ID displays that node along with its connected entities. Shown information includes node names, categories, and relationship labels. The system supports minor spelling errors and provides real-time entity suggestions in the search box as the user types, enabling intuitive and efficient exploration.   
### Data Dashboard Module  
This module presents real-time statistics from the local database through interactive visualizations—including bar charts, pie charts, and word clouds—to help users understand data composition and cybersecurity trends. Key metrics such as frequent keywords, vulnerability distributions, and APT organization characteristics are highlighted.  
### Intelligent QA Module  
Powered by the DeepSeek-V3 large language model, RAG (Retrieval-Augmented Generation), and prompt engineering, this module exploits the graph structure of the three domains for accurate retrieval and reasoning. It incorporates multiple fuzzy matching techniques—such as Levenshtein distance and word embedding similarity—to tolerate minor input errors. Users can ask questions about any entity, attribute, or relationship across the three cybersecurity domains, including single-entity lookups, multi-hop association queries, and inferential (asynchronous) reasoning. For instance: “Attack pattern A leads to outcome B—what other attack patterns could produce the same result?” Even with incomplete or slightly misspelled inputs, the system extracts relevant entities and relationships, then responds in natural language with high reliability and efficiency.   
   
## Knowledge Graph Description  
Data on cyber attack workflows, cybersecurity vulnerabilities, and nation-state Advanced Persistent Threat (APT) groups were collected from three distinct sources—via direct downloads, web crawling, and data integration—yielding nearly 20,000 raw records. These records were systematically decomposed into entities, relationships, and attributes to align with graph database requirements. Rigorous data cleaning—including handling missing values, removing outliers, and deduplication—ensured high data quality and validity. From an initial set of ~6,000 nodes and over 10,000 relationships, we refined the dataset into 4,237 entities across 20 types, over 7,000 relationships across 14 types, and 13 associated attributes, ensuring comprehensive coverage.  
<img width="220" height="460" alt="image" src="https://github.com/user-attachments/assets/f5df5a3b-3261-4325-bf97-4e9b9df6a5e3" /><img width="575" height="461" alt="image" src="https://github.com/user-attachments/assets/59418cdc-2e6f-4b8e-8edf-44d2ccb15f19" />  
To optimize for graph-based storage and querying, we designed three domain-specific graph schemas tailored to future search and reasoning needs. In the structure diagrams below, nodes of the same color belong to the same category; arrows indicate relationships; text inside circular nodes denotes entity types, and labels on arrows specify relationship types.  
### Attack Workflow Patterns  
By integrating domestic and international attack pattern data, we constructed a custom graph schema that captures the continuity of cyber operations—for instance, representing how one attack group may be followed by, collaborate with, or spawn subgroups. Each attack technique is linked to its prerequisites, indicators, and resulting impacts, providing actionable insights for defense and mitigation. The graph’s topology—where multiple groups may lead to the same outcome or share common prerequisites—naturally supports high-recall and multi-hop queries.  
<img width="468" height="401" alt="image" src="https://github.com/user-attachments/assets/57b5de21-99aa-42d0-9c6b-ca5c1ef59890" />  
<img width="459" height="341" alt="image" src="https://github.com/user-attachments/assets/3147230c-4eee-4bd0-b6a5-175aa32d74a9" />  
### Cybersecurity Vulnerabilities  
We gathered nearly a decade of vulnerability records from China’s National Vulnerability Database (CNNVD), including fields such as name, CNNVD ID, vulnerability type, severity level, and detailed descriptions. A dedicated graph structure was designed to enable efficient exploration—e.g., finding all vulnerabilities of a given type or comparing severity levels.  
<img width="308" height="256" alt="image" src="https://github.com/user-attachments/assets/80e23915-3a07-4883-ba14-91cbafd9f7c1" />  
<img width="426" height="457" alt="image" src="https://github.com/user-attachments/assets/ea598f49-1dfe-45cc-873e-014f7bc02b09" />  
<img width="438" height="351" alt="image" src="https://github.com/user-attachments/assets/b8fabcf4-19b4-4591-9838-e4d0eabb64e0" />  
### APT Organization Profiles  
We curated and integrated data on 74 global APT groups (including 51 major ones) from public reports, forums, and threat intelligence databases. Our graph schema emphasizes high connectivity and low sparsity, enabling in-depth analysis of inter-group relationships. Attributes such as organization type, tactics, techniques, and tools are stored as node properties to support rich semantic queries.  
<img width="723" height="431" alt="image" src="https://github.com/user-attachments/assets/a32d075b-9547-417d-8898-150e1fa47d89" />  
<img width="269" height="196" alt="image" src="https://github.com/user-attachments/assets/84525aa4-b970-42e9-be53-1c936dd9124c" />  
<img width="735" height="604" alt="image" src="https://github.com/user-attachments/assets/faa195fb-3730-4875-85a4-84d4543ffbac" />  

## Function Example  
<img width="375" height="165" alt="image" src="https://github.com/user-attachments/assets/d050098b-f5d6-4662-9624-74553057f729" />  
<img width="609" height="452" alt="image" src="https://github.com/user-attachments/assets/953d0021-240f-4fe1-987b-57117b7c4ea3" />  
<img width="717" height="393" alt="image" src="https://github.com/user-attachments/assets/7a525628-aa64-4ef8-8f9c-7c45a8a981c3" />  
<img width="723" height="401" alt="image" src="https://github.com/user-attachments/assets/c67ac4b0-5bd3-4acb-8d9e-a0f14f8cc4c1" />  

## Algorithm  
1. Initialization and Configuration  
The system first initializes the SentenceTransformer model, prioritizing GPU acceleration for multilingual (especially Chinese) processing support. It also connects to a Neo4j graph database and sets up an ARK API client to access large language models. Key parameters are configured during this phase, including fuzzy matching thresholds and vector similarity thresholds, which directly impact the accuracy of entity linking and node filtering.   
 
2. Fuzzy Entity Matching Process  
After receiving user queries, the system invokes the LLM to analyze the content, identifying key entities and their types (e.g., "attackpattern", "skill"). The LLM checks if there are any spelling errors in identified entities; if so, it returns them in the format "original_entity|corrected_entity". For example, input "sql 注人" might be recognized and corrected to "sql 注人|sql 注入". This process outputs structured JSON data to ensure clarity for subsequent processing. This step ensures robustness against user inputs, especially technical terms or multilingual entries, by outputting a structured list of entities ready for matching with the knowledge graph’s entity dictionary.  
Entity matching is implemented in the find_closest_pattern function using a two-stage strategy:  
**Levenshtein Distance Matching**: Initially calculates the edit distance between user input and entities in the dictionary. If the input contains a corrected entity separated by "|", the system computes matching scores for both the original and corrected entities, taking the higher score. If the highest score exceeds the threshold (default 81), that entity is returned.  
**Vector Embedding Matching**: If Levenshtein matching fails, the system uses the embedding_matching function to compute semantic similarity. When corrected entities are present, the combination "corrected_entity | original_entity" is used for embedding to preserve original meaning. The system loads the corresponding type of entity embeddings (e.g., "attackpattern_embeddings.txt") and compares the cosine similarity between the recognized entity's embedding vector and those in the dictionary. If the highest score surpasses the threshold (default 0.66), the entity is returned. This ensures non-standard expressions can accurately link to the standard knowledge framework.  
  
3. Prior Knowledge and Entity Pruning  
The system initially calls the LLM via the handle6 function to generate an initial response based on prior knowledge and obtain its embedding vector. This initial response is used for subsequent similarity pruning to filter out nodes with low semantic relevance to the user's question. In the add_node_to_cache_with_pruning function, new nodes undergo vector similarity calculation with the initial response; if the similarity is below the set threshold, the node is pruned and not added to the exploration path, effectively reducing the search space.  
  
4. Multi-hop Exploration Process  
Implemented within the iterative query loop of the handle6 function:  
Starting from the initial entity, each round selects the most relevant nodes (up to max_nodes) and retrieves their one-hop neighbors using the get_node_neighbors function, formatting neighbor information via format_neighbor_results_for_llm.  
Subsequent pruning through add_node_to_cache_with_pruning retains nodes with cosine similarities above the threshold in the node cache while marking nodes that have had their one-hop neighbors queried and acquiring community information to enhance contextual understanding.  
The LLM determines whether sufficient information has been gathered or if further querying is needed.  
If more information is required, the next batch of nodes is selected, generating new sub-questions.  
This process executes up to max_cypher times (default 3), controlling the depth of multi-hop exploration.  
The system maintains three critical caches:  
**Node Cache (node_cache)**: {node_name: {"summary": {...}, "is_valuable": 0/1}}, storing node summaries and query flags.  
**Community Cache (community_cache)**: {community_id: description}, storing community descriptions.  
**QA Cache (qa_cache)**: {node_name: {"question": q, "answer": a}}, storing sub-questions and answers related to nodes.  
These caches prevent redundant queries and calculations, providing global context for the LLM to understand graph structure relationships.  
  
5. Community Summary  
Through the get_community_info function, the system acquires community information for nodes, including community ID and description. This information helps the LLM understand the node's position and relationships within larger communities, particularly useful for densely connected graphs where community structures provide valuable clustering information, enhancing the understanding of node relationships and guiding deep searches from a global perspective.  
community node example:  
<img width="889" height="567" alt="image" src="https://github.com/user-attachments/assets/4a6ed406-d7d1-49e7-a593-d6b95091e4cc" />  

6. Sub-question Decomposition  
In the llm_judge_and_select_with_subq function, the LLM generates "intentional sub-questions" for each newly selected node, specifying:  
**Why was this node chosen?**  
**What information is expected from this node?**  
**How will this information help answer the final question?**  
After each round of queries, the system generates answers for previous sub-questions based on acquired results, forming QA pairs stored in qa_cache, building a logical chain of knowledge for exploring paths and aiding in generating the final answer.
Decomposing subproblem procedural steps:
<img width="946" height="221" alt="image" src="https://github.com/user-attachments/assets/c969c307-bc77-425d-bfa0-72f4c9862185" />

8. LLM Prompt Design  
The system provides highly structured information to the LLM, primarily including:  
**Node Summaries**: Contain basic information like labels, community IDs, and parent node relationships, providing graph structure details.  
**Community Descriptions**: Offer higher-level context about the node's community, guiding queries.  
**Historical QA Pairs**: Display questions and answers from explored nodes to ensure the original question is not lost or forgotten during deeper queries.  
**Current Query Results**: Include detailed neighbor relationship information from the latest query nodes, aiding in answering sub-questions and further addressing the original question.  
**Optional Node List**: Provides unexplored candidate nodes for selection, helping control the LLM's deep queries and avoid redundant node exploration. This information enables the LLM to: (1) Determine if current information is sufficient to answer the user's question; (2) Generate answers for sub-questions based on existing information; (3) Select the most relevant new nodes for continued querying; (4) Generate the final answer.
  
9. Formatting and Post-processing  
The system uses the format_neighbor_results_for_llm function to convert Neo4j query results into LLM-friendly JSON formats, filtering out redundant information and highlighting core relationships. After generating the final answer, additional formatting ensures a fluent, clearly structured natural language response.  

No predefined question formats or content restrictions are required—users can interact through natural language throughout the conversation. The system supports single-entity queries, fuzzy matching queries, and multi-step reasoning queries, and comprehensively handles queries across all categories of entities, attributes, and relationships.  
<img width="347" height="366" alt="image" src="https://github.com/user-attachments/assets/98ee6529-3dd3-4ab4-b194-45aafe0a89d2" />  
The corresponding node information in the database for the above example query:  
Below is the base node (i.e., the starting node for reasoning in the question):  
<img width="550" height="352" alt="image" src="https://github.com/user-attachments/assets/5ea81173-d3c3-4321-928d-ef00e65efacb" />  
Below is the target node (i.e., the node containing the information that should appear in the answer):  
<img width="558" height="359" alt="image" src="https://github.com/user-attachments/assets/3b40b1ce-057b-49d9-a256-9b81322c1fe9" />  










