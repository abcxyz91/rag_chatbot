# Multi-Agent RAG Chatbot với CrewAI

Một chatbot RAG (Retrieval-Augmented Generation) mạnh mẽ, được định tuyến động (dynamically-routed), xây dựng bằng [CrewAI](https://crewai.com/). Hệ thống này sử dụng framework điều phối đa tác nhân (multi-agent orchestration) để phân loại các truy vấn của người dùng và định tuyến chúng một cách liền mạch đến các chuyên gia AI theo từng lĩnh vực cụ thể (Kế toán, Nhân sự, Pháp lý hoặc Tổng hợp), đảm bảo các phản hồi có độ chính xác cao và phù hợp với ngữ cảnh.

Được thiết kế hướng tới hạ tầng cục bộ (local infrastructure), backend hỗ trợ thực thi nội bộ (local execution) bằng **Ollama** và **ChromaDB**. Điều này giúp dự án trở nên hoàn hảo cho các môi trường bảo mật, hiệu suất cao mà không cần phụ thuộc vào các API bên ngoài.

## 🌟 Tính năng chính

* **Định tuyến Ý định Thông minh (Intelligent Intent Routing):** Sử dụng một LLM nhanh và mang tính quyết định (`gemma3:4b`) để phân loại chính xác các truy vấn đầu vào của người dùng vào các lĩnh vực riêng biệt.
* **Các Nhóm Đa tác nhân theo Lĩnh vực (Domain-Specific Multi-Agent Crews):** 
  * 📊 **Nhóm Kế toán (Accounting Crew):** Truy xuất và phân tích các tài liệu tài chính, sổ cái và hồ sơ csv.
  * 🧑‍💼 **Nhóm Nhân sự (HR Crew):** Tra cứu các tài liệu chính sách nội bộ, tài liệu đào tạo và hướng dẫn.
  * ⚖️ **Nhóm Pháp lý (Legal Crew):** Phân tích các khuôn khổ pháp lý, hợp đồng và tài liệu tuân thủ.
  * 🌐 **Nhóm Tổng hợp (General Crew):** Xử lý các câu hỏi ngoài phạm vi một cách tinh tế bằng khả năng suy luận tổng quát.
* **Ngăn xếp AI Cục bộ, Ưu tiên Bảo mật (Local, Privacy-First AI Stack):** Tích hợp hoàn toàn với Ollama để tạo văn bản/embeddings và ChromaDB để lưu trữ vector, đảm bảo các tài liệu nội bộ của công ty bạn không bao giờ rời khỏi máy chủ.
* **Giao diện Streaming Thời gian thực:** Tích hợp giao diện web phản hồi nhanh được xây dựng bằng **Streamlit**, với tính năng streaming token theo thời gian thực để mang lại trải nghiệm chat tự nhiên.
* **Tích hợp tính năng OCR:** Các công cụ được tích hợp sẵn để trích xuất văn bản từ hình ảnh (ví dụ: biên lai được quét) để đưa vào luồng xử lý RAG.

## 🏗️ Kiến trúc

1. **Đầu vào của Người dùng (User Input):** Nhận thông qua CLI hoặc giao diện web Streamlit.
2. **Node Phân loại (Classification Node):** Một LLM nhẹ đánh giá prompt và trả về một phân loại JSON có kiểu dữ liệu nghiêm ngặt (strictly typed) sử dụng Pydantic.
3. **Node Định tuyến (Router Node):** Chuyển hướng truy vấn đến `Crew` chuyên trách tương ứng.
4. **Thực thi & RAG:** Nhóm được nhắm mục tiêu sử dụng các công cụ như `DirectoryReadTool`, `PDFSearchTool`, `DOCXSearchTool`, và `CSVSearchTool` để trích xuất ngữ cảnh liên quan từ ChromaDB.
5. **Tạo Phản hồi (Response Generation):** Một LLM lớn hơn, có khả năng xử lý mạnh hơn (`gemma3:27b`) tổng hợp câu trả lời cuối cùng cùng với các trích dẫn nguồn và stream ngược lại cho người dùng.

## 🛠️ Yêu cầu hệ thống

Trước khi bắt đầu, hãy đảm bảo bạn đã cài đặt các công cụ sau:
* **Python** >= 3.10, < 3.14
* **Ollama**: Đã cài đặt và đang chạy cục bộ. Bạn sẽ cần tải (pull) các model sau:
  ```bash
  ollama run gemma3:4b
  ollama run gemma3:27b
  ollama pull embeddinggemma:300m
  ```


## 📦 Cài đặt

1.  **Clone repository:**

    ```bash
    git clone <your-repo-url>
    cd rag_chatbot
    ```

2.  **Cài đặt các thư viện phụ thuộc:**
    Dự án sử dụng `pyproject.toml` để quản lý thư viện. Hãy cài đặt package và các yêu cầu của nó:

    ```bash
    pip install -e .
    ```

    *Lưu ý: Lệnh này sẽ cài đặt `crewai[tools]` và các package cần thiết khác.*

3.  **Thiết lập biến môi trường:**
    Tạo file `.env` tại thư mục gốc (nếu bạn muốn ghi đè các giá trị mặc định):

    ```env
    CHROMA_PATH=./chroma_db
    OLLAMA_BASE_URL=http://localhost:11434
    EMBED_MODEL=embeddinggemma:300m
    ```

4.  **Nạp Dữ liệu vào Cơ sở Tri thức (Knowledge Base):**
    Đặt các tài liệu theo lĩnh vực của bạn (`.pdf`, `.docx`, `.csv`) vào các thư mục tương ứng trong `src/rag_chatbot/knowledge_base/`.

## 🚀 Hướng dẫn Sử dụng

### 1\. Giao diện Web (Streamlit)

Để có trải nghiệm tương tác tốt nhất với real-time streaming, hãy chạy ứng dụng Streamlit:

```bash
streamlit run src/rag_chatbot/streamlit_app.py
```

### 2\. Giao diện Dòng lệnh (CLI)

Để tương tác trực tiếp với chatbot qua terminal:

```bash
kickoff
# HOẶC
python -m rag_chatbot.main
```

### 3\. Chạy công cụ trích xuất OCR độc lập

Để trích xuất văn bản từ hình ảnh (như biên lai) trước khi index:

```bash
python src/rag_chatbot/ocr_tool.py
```

## 📂 Cấu trúc Dự án

```text
rag_chatbot/
├── pyproject.toml              # Metadata dự án và các thư viện phụ thuộc
├── src/
│   └── rag_chatbot/
│       ├── main.py             # Điều phối Flow lõi, định tuyến và quản lý trạng thái
│       ├── streamlit_app.py    # Giao diện Streamlit với streaming callbacks
│       ├── streaming_listener.py # Thread-safe queue cho LLM token streaming
│       ├── ocr_tool.py         # Cài đặt vision-agent cho xử lý ảnh sang văn bản
│       ├── crews/              # Các nhóm đa tác nhân (crews) theo từng lĩnh vực
│       │   ├── accounting_crew/
│       │   ├── general_crew/
│       │   ├── hr_crew/
│       │   └── legal_crew/
│       │       # (Mỗi crew chứa các file agents.yaml, tasks.yaml, và định nghĩa crew riêng)
│       ├── knowledge_base/     # Các tài liệu nguồn phục vụ RAG
│       └── tools/              # Định nghĩa các custom tool
└── .gitignore
```

## 🧠 Quản lý Trạng thái (State Management)

Dự án sử dụng `Flow` của CrewAI và `BaseModel` của Pydantic để duy trì các kiểu dữ liệu nghiêm ngặt và lưu trữ lịch sử trò chuyện. Lịch sử sẽ tự động cắt bớt sau 20 tin nhắn để quản lý hiệu quả giới hạn ngữ cảnh (context windows) khi giao tiếp với các model Ollama cục bộ.

## 📄 Giấy phép (License)

Dự án này được cấp phép theo Giấy phép MIT - xem file LICENSE để biết thêm chi tiết.



# Multi-Agent RAG Chatbot with CrewAI

A powerful, dynamically-routed Retrieval-Augmented Generation (RAG) chatbot built with [CrewAI](https://crewai.com/). This system utilizes a multi-agent orchestration framework to classify user queries and seamlessly route them to domain-specific AI experts (Accounting, HR, Legal, or General), ensuring highly accurate and contextually relevant responses.

Designed with local infrastructure in mind, the backend natively supports local execution using **Ollama** and **ChromaDB**, making it an excellent fit for secure, high-performance environments without relying on external APIs.

## 🌟 Key Features

* **Intelligent Intent Routing:** Utilizes a fast, deterministic LLM (`gemma3:4b`) to accurately classify incoming user queries into discrete domains.
* **Domain-Specific Multi-Agent Crews:** * 📊 **Accounting Crew:** Retrieves and analyzes financial documents, ledgers, and csv records.
  * 🧑‍💼 **HR Crew:** Consults internal policy documents, training materials, and guidelines.
  * ⚖️ **Legal Crew:** Analyzes regulatory frameworks, contracts, and compliance docs.
  * 🌐 **General Crew:** Handles out-of-scope inquiries gracefully with general reasoning.
* **Local, Privacy-First AI Stack:** Fully integrated with Ollama for text generation/embeddings and ChromaDB for vector storage, ensuring your private company documents never leave your machine.
* **Real-time Streaming UI:** Includes a responsive web interface built with **Streamlit**, featuring real-time token streaming for a native chat experience.
* **Integrated OCR Capabilities:** Built-in tools for extracting text from images (e.g., scanned receipts) to feed into the RAG pipeline.

## 🏗️ Architecture

1. **User Input:** Received via CLI or the Streamlit web interface.
2. **Classification Node:** A lightweight LLM evaluates the prompt and outputs a strictly typed JSON classification using Pydantic.
3. **Router Node:** Directs the query to the corresponding specialized `Crew`.
4. **Execution & RAG:** The targeted crew uses tools like `DirectoryReadTool`, `PDFSearchTool`, `DOCXSearchTool`, and `CSVSearchTool` to pull relevant context from ChromaDB.
5. **Response Generation:** A larger, more capable LLM (`gemma3:27b`) synthesizes the final answer with source citations and streams it back to the user.

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:
* **Python** >= 3.10, < 3.14
* **Ollama**: Installed and running locally. You will need to pull the following models:
  ```bash
  ollama run gemma3:4b
  ollama run gemma3:27b
  ollama pull embeddinggemma:300m
  ```


## 📦 Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd rag_chatbot
    ```

2.  **Install dependencies:**
    The project uses `pyproject.toml` for dependency management. Install the package and its requirements:

    ```bash
    pip install -e .
    ```

    *Note: This will install `crewai[tools]` and other required packages.*

3.  **Set up environment variables:**
    Create a `.env` file in the root directory (if you wish to override defaults):

    ```env
    CHROMA_PATH=./chroma_db
    OLLAMA_BASE_URL=http://localhost:11434
    EMBED_MODEL=embeddinggemma:300m
    ```

4.  **Populate Knowledge Base:**
    Place your domain-specific documents (`.pdf`, `.docx`, `.csv`) into their respective folders under `src/rag_chatbot/knowledge_base/`.

## 🚀 Usage

### 1\. Web Interface (Streamlit)

For the best interactive experience with real-time streaming, run the Streamlit app:

```bash
streamlit run src/rag_chatbot/streamlit_app.py
```

### 2\. Command Line Interface

To interact with the chatbot directly via the terminal:

```bash
kickoff
# OR
python -m rag_chatbot.main
```

### 3\. Running OCR Extraction standalone

To extract text from images (like receipts) before indexing:

```bash
python src/rag_chatbot/ocr_tool.py
```

## 📂 Project Structure

```text
rag_chatbot/
├── pyproject.toml              # Project metadata and dependencies
├── src/
│   └── rag_chatbot/
│       ├── main.py             # Core Flow orchestration, routing, and state management
│       ├── streamlit_app.py    # Streamlit frontend with streaming callbacks
│       ├── streaming_listener.py # Thread-safe queue for LLM token streaming
│       ├── ocr_tool.py         # Vision-agent setup for image-to-text processing
│       ├── crews/              # Domain-specific multi-agent crews
│       │   ├── accounting_crew/
│       │   ├── general_crew/
│       │   ├── hr_crew/
│       │   └── legal_crew/
│       │       # (Each crew contains its own agents.yaml, tasks.yaml, and crew definition)
│       ├── knowledge_base/     # Source documents for RAG
│       └── tools/              # Custom tool definitions
└── .gitignore
```

## 🧠 State Management

The project utilizes CrewAI's `Flow` and Pydantic `BaseModel` to maintain strict types and persist conversation history. The history automatically truncates after 20 messages to manage context windows efficiently when communicating with local Ollama models.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
