# AAG Data Pipeline

This module manages data acquisition and transformation within the AAG system.

---

## 📡 knowledge_ingestion/
**Purpose:**  
Collects external data from online sources such as Arxiv, OpenAlex, or other APIs.  
This data is mainly used to build the **knowledge layer** of the hierarchical knowledge base.

**Main Functions:**
- Crawl or fetch papers, blogs, or technical documents.
- Extract metadata (title, author, abstract, etc.).
- Save raw and structured outputs for further processing.

---

## 🔄 data_transformer/
**Purpose:**  
Processes and converts user-uploaded data into structured formats (e.g., graph data).  
The output is used by the **computing engine** for analysis and algorithm execution.

**Main Functions:**
- Parse user input files (CSV, JSON, GraphML, etc.).
- Extract entities and relationships.
- Build graph schemas (nodes, edges) and load into the graph database.

---

## Directory Summary
