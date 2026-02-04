# 研究助手Agent

## 1. 项目概述

### 1.1 项目目标

构建一个能够辅助学术研究的多功能Agent，具备论文搜索、阅读理解、文献综述生成等功能。

### 1.2 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                   研究助手Agent                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              用户交互层                           │   │
│  │    Web界面 / API / 命令行 / 微信小程序            │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │              任务规划层                           │   │
│  │      任务分解 → 步骤规划 → 执行监控 → 结果整合    │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │              工具层                               │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │   │
│  │  │论文搜索 │ │论文阅读 │ │翻译工具 │ │写作助手 │  │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │              知识管理层                          │   │
│  │      文献库管理 → 知识图谱 → 向量检索 → 笔记系统  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 2. 核心功能

### 2.1 论文搜索

```python
import requests
from scholarly import scholarly

class PaperSearch:
    """论文搜索工具"""

    def search_arxiv(self, query: str, max_results: int = 10) -> List[Dict]:
        """ArXiv搜索"""
        import arxiv

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        return [
            {
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.summary,
                "pdf_url": paper.pdf_url,
                "published": paper.published.strftime("%Y-%m-%d"),
                "categories": paper.categories
            }
            for paper in search.results()
        ]

    def search_semantic_scholar(self, query: str, limit: int = 10) -> List[Dict]:
        """Semantic Scholar搜索"""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,abstract,year,citationCount,url"
        }

        response = requests.get(url, params=params)
        data = response.json()

        return [
            {
                "title": p["title"],
                "authors": [a["name"] for a in p.get("authors", [])],
                "abstract": p.get("abstract", ""),
                "year": p.get("year"),
                "citations": p.get("citationCount", 0),
                "url": p.get("url")
            }
            for p in data.get("data", [])
        ]

    def search_google_scholar(self, query: str) -> List[Dict]:
        """Google Scholar搜索（使用scholarly库）"""
        search = scholarly.search_pubs(query)

        results = []
        for i, paper in enumerate(search):
            if i >= 10:
                break

            results.append({
                "title": paper.get("bib", {}).get("title"),
                "authors": paper.get("bib", {}).get("author"),
                "year": paper.get("bib", {}).get("pub_year"),
                "citations": paper.get("num_citations", 0),
                "venue": paper.get("bib", {}).get("venue")
            })

        return results
```

### 2.2 论文阅读

```python
from PyPDF2 import PdfReader
import pdfplumber

class PaperReader:
    """论文阅读器"""

    def __init__(self, llm):
        self.llm = llm

    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """提取PDF文本"""
        text = ""

        # 使用pdfplumber提取（保留布局信息）
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"

        return {
            "full_text": text,
            "num_pages": len(text) // 2000 + 1
        }

    def extract_sections(self, text: str) -> Dict:
        """提取论文各章节"""
        import re

        sections = {}
        current_section = "introduction"

        for line in text.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.\s+', line):  # 新章节
                match = re.match(r'^\d+\.\s+([A-Za-z\s]+)', line)
                if match:
                    current_section = match.group(1).lower().replace(' ', '_')
            sections[current_section] = sections.get(current_section, "") + line + "\n"

        return sections

    def summarize_section(self, section_text: str, section_name: str) -> str:
        """摘要章节"""
        prompt = f"""
请用100字以内摘要以下{section_name}部分的内容：

{section_text[:2000]}

摘要：
"""
        return self.llm.generate(prompt, max_tokens=150)

    def generate_paper_summary(self, pdf_path: str) -> Dict:
        """生成论文摘要"""
        # 提取文本
        text_data = self.extract_text_from_pdf(pdf_path)

        # 提取章节
        sections = self.extract_sections(text_data["full_text"])

        # 摘要各章节
        summary = {
            "基本信息": {},
            "各章节摘要": {}
        }

        for name, text in sections.items():
            if len(text) > 100:  # 跳过空章节
                summary["各章节摘要"][name] = self.summarize_section(text, name)

        # 整体摘要
        summary["整体摘要"] = self.summarize_section(
            text_data["full_text"],
            "全文"
        )

        return summary
```

### 2.3 文献综述生成

```python
class LiteratureReviewGenerator:
    """文献综述生成器"""

    def __init__(self, llm):
        self.llm = llm

    def generate_review(
        self,
        topic: str,
        papers: List[Dict],
        structure: Dict = None
    ) -> str:
        """生成文献综述"""
        # 组织论文信息
        paper_info = self._organize_papers(papers)

        # 生成综述
        if structure:
            review = self._generate_structured_review(topic, paper_info, structure)
        else:
            review = self._generate_linear_review(topic, paper_info)

        return review

    def _organize_papers(self, papers: List[Dict]) -> List[Dict]:
        """组织论文信息"""
        organized = []
        for p in papers:
            organized.append({
                "标题": p.get("title", ""),
                "作者": ", ".join(p.get("authors", [])[:3]),
                "年份": p.get("year", ""),
                "摘要": p.get("abstract", "")[:300],
                "引用数": p.get("citations", 0)
            })
        return organized

    def _generate_structured_review(
        self,
        topic: str,
        papers: List[Dict],
        structure: Dict
    ) -> str:
        """按结构生成综述"""
        prompt = f"""
请为以下主题撰写文献综述：

主题：{topic}

论文信息：
{papers}

综述结构要求：
{chr(10).join([f'{k}: {v}' for k, v in structure.items()])}

请撰写一篇完整的学术文献综述，包含：
1. 引言（研究背景和重要性）
2. 各部分内容
3. 总结和未来方向

请使用学术写作风格，引用相关论文。
"""
        return self.llm.generate(prompt, max_tokens=2000)

    def _generate_linear_review(self, topic: str, papers: List[Dict]) -> str:
        """线性生成综述"""
        prompt = f"""
请为以下主题撰写文献综述：

主题：{topic}

综述以下{len(papers)}篇论文的研究内容、主要发现和贡献：

{papers}

请按以下格式撰写：
1. 引言（约200字）
2. 主要研究方向（约400字）
3. 关键发现（约400字）
4. 总结和展望（约200字）

共约1200字。
"""
        return self.llm.generate(prompt, max_tokens=1500)
```

### 2.4 完整研究助手Agent

```python
class ResearchAssistantAgent:
    """研究助手Agent"""

    def __init__(self, llm):
        self.llm = llm
        self.paper_search = PaperSearch()
        self.paper_reader = PaperReader(llm)
        self.literature_review = LiteratureReviewGenerator(llm)
        self.note_manager = NoteManager()

    def research_topic(self, topic: str, depth: str = "surface") -> Dict:
        """
        研究主题

        depth: surface (表面了解) / medium (中等深度) / deep (深度研究)
        """
        result = {
            "topic": topic,
            "papers": [],
            "summary": "",
            "review": "",
            "questions": []
        }

        # 1. 搜索相关论文
        papers = self.paper_search.search_semantic_scholar(topic, limit=10)
        result["papers"] = papers

        # 2. 分类整理
        categorized = self._categorize_papers(papers)

        # 3. 生成综述
        structure = {
            "引言": "研究背景和意义",
            "主要方法": "现有方法分类",
            "关键发现": "重要研究成果",
            "未来方向": "研究局限和展望"
        }
        result["review"] = self.literature_review.generate_review(
            topic, papers, structure
        )

        # 4. 生成待研究问题
        result["questions"] = self._generate_research_questions(topic, papers)

        return result

    def _categorize_papers(self, papers: List[Dict]) -> Dict:
        """分类论文"""
        # 调用LLM进行分类
        prompt = f"""
请将以下论文按研究主题分类：

{json.dumps(papers, ensure_ascii=False, indent=2)}

JSON格式返回：
{{
    "类别1": [论文索引列表],
    "类别2": [论文索引列表]
}}
"""
        result = self.llm.generate(prompt)
        return json.loads(result)

    def _generate_research_questions(
        self,
        topic: str,
        papers: List[Dict]
    ) -> List[str]:
        """生成研究问题"""
        prompt = f"""
基于以下{topic}相关论文，提出5个值得深入研究的问题：

{json.dumps(papers, ensure_ascii=False, indent=2)}

研究问题（每行一个）：
"""
        result = self.llm.generate(prompt)
        return [q.strip() for q in result.split('\n') if q.strip()]

    def chat_about_paper(self, paper_path: str, question: str) -> str:
        """关于论文对话"""
        # 读取论文
        summary = self.paper_reader.generate_paper_summary(paper_path)

        prompt = f"""
基于以下论文摘要，回答用户的问题：

论文摘要：
{json.dumps(summary, ensure_ascii=False, indent=2)}

用户问题：{question}

请基于论文内容回答：
"""
        return self.llm.generate(prompt)
```

### 2.5 笔记管理

```python
class NoteManager:
    """笔记管理器"""

    def __init__(self):
        self.notes: Dict[str, Dict] = {}
        self.vector_store = VectorStore()

    def add_note(
        self,
        note_id: str,
        content: str,
        paper_ref: str = None,
        tags: List[str] = None
    ):
        """添加笔记"""
        self.notes[note_id] = {
            "content": content,
            "paper_ref": paper_ref,
            "tags": tags or [],
            "created_at": datetime.now().isoformat()
        }

        # 存入向量库
        self.vector_store.add(content, {"id": note_id, "tags": tags})

    def search_notes(self, query: str) -> List[Dict]:
        """搜索笔记"""
        results = self.vector_store.search(query)
        return [self.notes.get(r["id"], {}) for r in results]
```

## 3. 界面集成

### 3.1 Gradio界面

```python
import gradio as gr

def create_gradio_interface(agent: ResearchAssistantAgent):
    """创建Gradio界面"""

    with gr.Blocks(title="研究助手") as interface:
        gr.Markdown("# 📚 研究助手Agent")

        with gr.Tab("主题研究"):
            topic_input = gr.Textbox(label="研究主题")
            depth_dropdown = gr.Dropdown(
                ["surface", "medium", "deep"],
                value="medium",
                label="研究深度"
            )
            research_btn = gr.Button("开始研究", variant="primary")
            research_output = gr.Markdown()

            research_btn.click(
                agent.research_topic,
                inputs=[topic_input, depth_dropdown],
                outputs=research_output
            )

        with gr.Tab("论文问答"):
            paper_input = gr.File(label="上传论文 (PDF)")
            question_input = gr.Textbox(label="问题")
            qa_btn = gr.Button("提问", variant="primary")
            qa_output = gr.Textbox(label="回答")

            qa_btn.click(
                agent.chat_about_paper,
                inputs=[paper_input, question_input],
                outputs=qa_output
            )

        with gr.Tab("文献笔记"):
            note_input = gr.Textbox(label="笔记内容")
            tags_input = gr.Textbox(label="标签 (逗号分隔)")
            add_note_btn = gr.Button("添加笔记")
            search_input = gr.Textbox(label="搜索笔记")
            search_output = gr.JSON()

    return interface.launch()
```

## 4. 项目评估

| 功能 | 评估指标 | 目标 |
|-----|---------|------|
| 论文搜索 | 召回率/相关性 | >80% |
| 内容摘要 | ROUGE分数 | >0.4 |
| 对话问答 | 准确率 | >75% |
| 综述生成 | 完整性/连贯性 | 人工评估 |

## 5. 扩展建议

- 添加论文推荐功能
- 集成引用管理
- 支持协作标注
- 生成可视化图表
- 导出不同格式
