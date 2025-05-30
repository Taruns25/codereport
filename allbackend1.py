from typing import List, Dict, Any, TypedDict
from tree_sitter import Parser, Language
from tree_sitter_languages import get_language
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
import ast
from reportlab.platypus import Paragraph, Spacer, Image, PageBreak, SimpleDocTemplate, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_LEFT
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer, get_lexer_for_filename
from langchain_core.output_parsers import StrOutputParser
from reportlab.lib.utils import ImageReader
from PIL import Image as PILImage
from math import ceil
import graphviz
import os
import re
from reportlab.lib import colors
from reportlab.lib.units import inch
from dotenv import load_dotenv
import traceback
import tempfile
import shutil
import subprocess
import stat

load_dotenv()

# ====================== NEW: EXTENSION MAPPING ======================
EXTENSION_MAP = {
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.py': 'python',
    '.java': 'java',
    '.c': 'c',
    '.h': 'c',
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.hpp': 'cpp',
    '.go': 'go',
    '.rs': 'rust',
    '.php': 'php',
    '.rb': 'ruby',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.cs': 'csharp'
}
# ====================================================================

def clone_github_repo(repo_url: str) -> str:
    """Clones a GitHub repository and returns the path to the local copy"""
    temp_dir = tempfile.mkdtemp(prefix="repo_")
    try:
        subprocess.run(['git', 'clone', repo_url, temp_dir], check=True)
        print(f"Cloned repository to: {temp_dir}")
        return temp_dir
    except subprocess.CalledProcessError as e:
        shutil.rmtree(temp_dir)
        raise RuntimeError(f"Failed to clone GitHub repo: {e}")

def handle_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def process_input_path(input_path: str, include_uml=True):
    """Accepts either a local path or GitHub URL, processes code files"""
    is_url = input_path.startswith("http://") or input_path.startswith("https://")
    local_path = input_path
    
    if is_url:
        if "github.com" in input_path:
            local_path = clone_github_repo(input_path)
        else:
            raise ValueError("Only GitHub URLs are supported currently.")

    process_directory(local_path, include_uml=include_uml)

    if is_url:
        shutil.rmtree(local_path, onerror=handle_remove_readonly)
        print(f"Cleaned up temporary clone: {local_path}")
    
# Universal file reader with encoding fallback
def read_file_with_fallback(filepath):
    encodings = ['utf-8-sig', 'utf-16', 'utf-8', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
                # Skip obviously invalid Python files
                if content.strip() == "" or len(content) < 10:
                    raise ValueError("File is empty or too short.")
                return content
        except (UnicodeDecodeError, ValueError):
            continue
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return ""
    return ""

class FileState(TypedDict):
    file_path: str
    content: str
    language: str
    code_blocks: List[Dict[str, Any]]
    summaries: Dict[str, str]
    screenshots: Dict[str, str]
    pdf_path: str
    uml_diagram: str
    dir_path: str

llm = AzureChatOpenAI(
    temperature=0.7,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-05-15",
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT")
)

# Enhanced language configuration
LANGUAGE_CONFIG = {
    'python': {
        'ast_parser': True,
        'tree_sitter_query': """
            (class_definition name: (identifier) @name) @block_node
            (function_definition name: (identifier) @name) @block_node
        """
    },
    'javascript': {
        'tree_sitter_query': """
            (function_declaration name: (identifier) @name) @block_node
            (class_declaration name: (identifier) @name) @block_node
            (lexical_declaration (variable_declarator name: (identifier) value: (arrow_function)) @block_node)
            (variable_declarator name: (identifier) value: [(function) (arrow_function)]) @block_node
        """
    },
    'java': {
        'tree_sitter_query': """
            (class_declaration name: (identifier) @name) @block_node
            (method_declaration name: (identifier) @name) @block_node
        """
    },
    'c': {
        'tree_sitter_query': """
            (function_definition declarator: (function_declarator declarator: (identifier)) @name) @block_node
            (struct_specifier name: (type_identifier) @name) @block_node
        """
    },
    'cpp': {
        'tree_sitter_query': """
            (function_definition declarator: (function_declarator declarator: (identifier)) @name) @block_node
            (class_specifier name: (identifier) @name) @block_node
            (template_declaration body: (class_specifier name: (identifier) @name)) @block_node
            (namespace_definition name: (identifier) @name) @block_node
        """
    },
    'go': {
        'tree_sitter_query': """
            (function_declaration name: (identifier) @name) @block_node
            (type_declaration (type_spec name: (type_identifier) @name body: (struct_type)) @block_node
        """
    },
    'rust': {
        'tree_sitter_query': """
            (function_item name: (identifier) @name) @block_node
            (struct_item name: (type_identifier) @name) @block_node
            (impl_item (type_identifier) @name) @block_node
        """
    },
    'typescript': {
        'tree_sitter_query': """
            (function_declaration name: (identifier) @name) @block_node
            (class_declaration name: (identifier) @name) @block_node
            (interface_declaration name: (identifier) @name) @block_node
            (type_alias_declaration name: (identifier) @name) @block_node
        """
    },
    'default': {
        'tree_sitter_query': None
    }
}

# Global parser instances with improved error handling
PARSERS = {}
LANGUAGES_TO_LOAD = [
    'python', 'javascript', 'java', 'c', 'cpp', 'go', 'rust',
    'typescript', 'php', 'ruby', 'swift', 'kotlin'
]

print("Initializing Tree-sitter parsers...")
for lang_name in LANGUAGES_TO_LOAD:
    try:
        lang = get_language(lang_name)
        if lang:
            parser = Parser()
            parser.set_language(lang)
            PARSERS[lang_name] = (parser, lang)
            print(f"Loaded {lang_name} parser successfully")
        else:
            print(f"Language not available for {lang_name}")
            PARSERS[lang_name] = None
    except Exception as e:
        print(f"Error loading {lang_name}: {str(e)}")
        # Continue to load other parsers even if one fails
        PARSERS[lang_name] = None
        continue


# PDF styles initialization
PDF_STYLES = getSampleStyleSheet()
if 'MyBullet' not in PDF_STYLES:
    PDF_STYLES.add(ParagraphStyle(name='MyBullet', parent=PDF_STYLES['Normal'],
                              leftIndent=20, bulletIndent=10, bulletFontName='Symbol',
                              bulletFontSize=10, bulletOffsetY=-2))
if 'Code' not in PDF_STYLES:
    PDF_STYLES.add(ParagraphStyle(
        name='Code',
        fontName='Courier',
        fontSize=10,
        leading=12,
        textColor=colors.darkblue,
        backColor=colors.lightgrey,
        leftIndent=20,
        rightIndent=20,
        borderColor=colors.black,
        borderWidth=0.5,
        borderPadding=5
    ))

# ====================== UPDATED LANGUAGE DETECTION ======================
def detect_language(file_path: str, content: str) -> str:
    """Enhanced language detection with Pygments-compatible names"""
    # Try by file extension first
    ext = Path(file_path).suffix.lower()
    if ext in EXTENSION_MAP:
        return EXTENSION_MAP[ext]
    
    # Try Pygments filename detection
    try:
        lexer = get_lexer_for_filename(file_path)
        return lexer.aliases[0] if lexer.aliases else lexer.name.lower()
    except:
        pass
    
    # Try content-based guessing
    try:
        lexer = guess_lexer(content)
        return lexer.aliases[0] if lexer.aliases else lexer.name.lower()
    except:
        pass
    
    # Final fallback
    return 'text'
# ========================================================================

def analyze_full_code(state: FileState) -> FileState:
    try:
        state['content'] = read_file_with_fallback(state['file_path'])
        state['language'] = detect_language(state['file_path'], state['content'])
        print(f"Detected language: {state['language']} for file: {state['file_path']}")

        prompt = ChatPromptTemplate.from_template("""
        Analyze the following {language} code and provide a technical summary:
        {code}
        Focus on:
        - Main purpose and functionality
        - Key components and their interactions
        - Important design patterns
        - Notable dependencies
        - Language-specific considerations
        """)

        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"code": state['content'], "language": state['language']})
        state['summaries']['full_code'] = summary
    except Exception as e:
        state['summaries']['full_code'] = f"Analysis failed: {str(e)}"
    return state

def extract_code_blocks(state: FileState) -> FileState:
    state['code_blocks'] = []
    content = state['content']
    language = state['language']
    file_name = Path(state['file_path']).name  # Fixed path handling
    file_path = Path(state['file_path'])

    try:
        # Try Python AST first for Python files
        if language == 'python':
            try:
                tree = ast.parse(content)
                blocks = []

                for node in ast.walk(tree):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                        block_code = ast.get_source_segment(content, node) or "\n".join(
                            content.splitlines()[node.lineno - 1:node.end_lineno]
                        )
                        blocks.append({
                            'name': node.name,
                            'type': type(node).__name__.replace('Def', ''),
                            'start_line': node.lineno,
                            'end_line': node.end_lineno,
                            'code': block_code
                        })

                if blocks:
                    state['code_blocks'] = blocks
                    return state
            except Exception as e:
                print(f"AST parsing failed for {file_path.name}: {e}")
        
        # Try Tree-sitter parsing
        parser_info = PARSERS.get(language)
        if parser_info:
            parser, lang = parser_info
            query_str = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG['default'])['tree_sitter_query']
            
            if query_str:
                tree = parser.parse(content.encode('utf8'))
                query = lang.query(query_str)
                captures = query.captures(tree.root_node)

                blocks = []
                current_block = None
                for node, capture_name in captures:
                    if capture_name == 'block_node':
                        if current_block:
                            blocks.append(current_block)
                        current_block = {
                            'name': '',
                            'type': node.type,
                            'start_line': node.start_point[0]+1,
                            'end_line': node.end_point[0]+1,
                            'code': node.text.decode('utf8')
                        }
                        name_node = node.child_by_field_name('name')
                        if name_node:
                            current_block['name'] = name_node.text.decode('utf8')
                        else:
                            current_block['name'] = f"anon_{node.type}_{node.start_point[0]+1}"
                    elif capture_name == 'name' and current_block:
                        current_block['name'] = node.text.decode('utf8')
                
                if current_block:
                    blocks.append(current_block)
                
                if blocks:
                    state['code_blocks'] = blocks
                    return state

        # Fallback to regex-based parsing
        print(f"Using regex fallback for {file_name}")
        regex_patterns = {
            'python': r'^(?:class|def)\s+(\w+)\s*[:(]',
            'java': r'(?:public|private|protected)?\s*(?:static|final)?\s*(?:class|interface|enum)\s+(\w+)',
            'javascript': r'(?:function|class)\s+(\w+)',
            'c': r'^\w+\s+\*?(\w+)\s*\([^)]*\)\s*{',
            'default': r'(?:def|function|class|struct|interface)\s+(\w+)'
        }
        pattern = regex_patterns.get(language, regex_patterns['default'])
        blocks = []
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            block_name = match.group(1)
            block_type = 'function' if 'function' in match.group(0).lower() else 'class'
            blocks.append({
                'name': block_name,
                'type': block_type,
                'start_line': content[:match.start()].count('\n') + 1,
                'end_line': content[:match.end()].count('\n') + 1,
                'code': match.group(0)
            })
        state['code_blocks'] = blocks

    except Exception as e:
        print(f"Error extracting blocks from {file_name}: {str(e)}")
        state['code_blocks'] = []
    
    return state

def explain_code_with_llm(block, state: FileState) -> str:
    """Generate explanations for code blocks using LLM."""
    prompt = ChatPromptTemplate.from_template("""
    Explain this {language} code block in technical terms:
    {code}
    Focus on:
    - Purpose and functionality
    - Inputs/outputs
    - Architectural role
    - Key algorithms
    - Security considerations
    """)
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "code": block['code'],
        "language": state['language']
    })

# ====================== UPDATED SCREENSHOT GENERATION ======================
def generate_code_block_screenshot(state: FileState) -> FileState:
    for block in state['code_blocks']:
        block_name = block.get('name', f"block_{state['code_blocks'].index(block)}")
        try:
            # Get lexer with multiple fallbacks
            lexer = None
            language = state['language'].lower()
            
            # Try direct match
            try:
                lexer = get_lexer_by_name(language)
            except:
                # Try common aliases
                alias_map = {
                    'js': 'javascript',
                    'javascript': 'javascript',
                    'ts': 'typescript',
                    'typescript': 'typescript',
                    'py': 'python',
                    'python': 'python',
                    'c': 'c',
                    'cpp': 'cpp',
                    'c++': 'cpp',
                    'java': 'java',
                    'go': 'go',
                    'golang': 'go',
                    'rs': 'rust',
                    'rust': 'rust',
                    'text': 'text'
                }
                lexer_name = alias_map.get(language, 'text')
                lexer = get_lexer_by_name(lexer_name)
            
            # Create temp file in a cross-platform way
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_image_path = temp_file.name
            
            # Generate the image
            formatter = ImageFormatter(
                style='monokai',
                line_numbers=True,
                font_size=12,
                line_pad=0,
                image_pad=10,
                image_format='png'
            )
            
            highlight(block['code'], lexer, formatter, outfile=temp_image_path)
            
            # Resize if too wide
            with PILImage.open(temp_image_path) as img:
                if img.width > 1200:
                    ratio = 1200 / img.width
                    new_height = int(img.height * ratio)
                    img.resize((1200, new_height), PILImage.Resampling.LANCZOS).save(temp_image_path)
            
            state['screenshots'][block_name] = temp_image_path
            print(f"Generated screenshot for {block_name} at {temp_image_path}")
            
            # Generate summary if not already present
            if block_name not in state['summaries']:
                state['summaries'][block_name] = explain_code_with_llm(block, state)
                
        except Exception as e:
            print(f"Error generating screenshot for {block_name}: {str(e)}")
            traceback.print_exc()
            
            # Fallback: store code as text
            state['screenshots'][block_name] = None
            if block_name not in state['summaries']:
                state['summaries'][block_name] = f"Code block:\n{block['code']}"
    
    return state
# ===========================================================================

def split_image_for_pdf(image_path, max_height=500):
    with PILImage.open(image_path) as img:
        width, height = img.size
        if height <= max_height:
            return [image_path]

        n_sections = ceil(height / max_height)
        image_sections = []
        for i in range(n_sections):
            top = i * max_height
            bottom = min((i + 1) * max_height, height)
            section = img.crop((0, top, width, bottom))
            section_path = f"{image_path}_section_{i}.png"
            section.save(section_path)
            image_sections.append(section_path)
        return image_sections

def format_markdown_to_story(text: str, styles) -> list:
    """Convert markdown text to ReportLab story elements."""
    story = []
    lines = text.strip().split('\n')
    in_code_block = False
    current_code = []
    current_list = []

    for line in lines:
        stripped = line.strip()

        # Handle code blocks
        if stripped.startswith('```'):
            if in_code_block:
                story.append(Preformatted('\n'.join(current_code), styles['Code']))
                current_code = []
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            current_code.append(line)
            continue

        # Handle headers
        if stripped.startswith('###'):
            story.append(Paragraph(stripped[3:].strip(), styles['Heading3']))
        elif stripped.startswith('##'):
            story.append(Paragraph(stripped[2:].strip(), styles['Heading2']))
        elif stripped.startswith('#'):
            story.append(Paragraph(stripped[1:].strip(), styles['Heading1']))

        # Handle lists
        elif stripped.startswith('- '):
            if current_list:
                story.extend(current_list)
                current_list = []
            current_list.append(Paragraph(f"• {stripped[2:]}", styles['MyBullet']))

        # Handle paragraphs and empty lines
        else:
            if current_list:
                story.extend(current_list)
                current_list = []
            if stripped:
                story.append(Paragraph(stripped, styles['Normal']))
            else:
                story.append(Spacer(1, 12))

    # Add any remaining code/list items
    if in_code_block and current_code:
        story.append(Preformatted('\n'.join(current_code), styles['Code']))
    if current_list:
        story.extend(current_list)

    return story

# ====================== UPDATED PDF GENERATION ======================
def generate_pdf_report(state: FileState) -> FileState:
    """Generate the final PDF report with code analysis and visuals."""
    try:
        doc = SimpleDocTemplate(
            state['pdf_path'],
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        styles = PDF_STYLES
        story = []
        available_width = letter[0] - 144  # 72*2 margins
        available_height = letter[1] - 144  # for max image height in points
        max_image_height = 500 # Maximum height for images in points
        story.append(Paragraph("Code Overview", styles['Heading1']))

        # Add UML diagram
        if 'uml_diagram' in state and os.path.exists(state['uml_diagram']):
            try:
                story.append(Paragraph("Architecture Diagram", styles['Heading2']))
                img = ImageReader(state['uml_diagram'])
                img_width, img_height = img.getSize()
                scale_factor = min(available_width / img_width, max_image_height / img_height) # Scale based on both dimensions
                new_width = img_width * scale_factor
                new_height = img_height * scale_factor
                story.append(Image(state['uml_diagram'], width=new_width, height=new_height))
                story.append(Spacer(1, 24))
            except Exception as e:
                print(f"Error adding UML diagram: {str(e)}")

        # Add code analysis content
        story.append(Paragraph("Code Analysis Report", styles['Title']))
        story.append(Spacer(1, 12))
        story.extend(format_markdown_to_story(state['summaries']['full_code'], styles))
        story.append(PageBreak())

        # Add code blocks with screenshots
        story.append(Paragraph("Detailed Code Analysis", styles['Heading1']))
        for block in state['code_blocks']:
            block_name = block.get('name', f"block_{state['code_blocks'].index(block)}")
            story.append(Paragraph(f"{block['type'].title()}: {block_name}", styles['Heading2']))
            
            # Handle screenshot if available
            screenshot_path = state['screenshots'].get(block_name)
            if screenshot_path and os.path.exists(screenshot_path):
                try:
                    sections = split_image_for_pdf(screenshot_path, max_height=available_height)

                    for section in sections:
                        with PILImage.open(section) as img:
                            scale = available_width / img.width
                            story.append(Image(section,
                                               width=img.width * scale,
                                               height=img.height * scale))
                        story.append(PageBreak())
                except Exception as e:
                    print(f"Error adding code screenshot: {str(e)}")
                    story.append(Paragraph("Could not generate screenshot", styles['Italic']))
            else:
                # Add code as text if screenshot missing
                story.append(Paragraph("Code:", styles['Heading3']))
                story.append(Preformatted(block['code'], styles['Code']))
            
            # Add summary
            if block_name in state['summaries']:
                story.extend(format_markdown_to_story(state['summaries'][block_name], styles))
            
            story.append(Spacer(1, 24))
            story.append(PageBreak())
        
        doc.build(story)
        print(f"PDF report generated at: {state['pdf_path']}")

        # Enhanced cleanup
        for block_name, screenshot_path in state['screenshots'].items():
            if screenshot_path and os.path.exists(screenshot_path):
                try:
                    os.unlink(screenshot_path)
                    # Also clean up split sections if any
                    base_path = Path(screenshot_path).stem
                    for section_file in Path(".").glob(f"{base_path}_section_*.png"):
                        try:
                            section_file.unlink()
                        except:
                            pass
                except Exception as e:
                    print(f"Could not delete {screenshot_path}: {e}")

    except Exception as e:
        print(f"PDF generation failed: {str(e)}")
        traceback.print_exc()

    return state
# ====================================================================


class FileAnalyzer:
    def __init__(self, root_path):
        self.root_path = Path(root_path).resolve()
        # Changed rankdir to 'TB' as per user's preference
        self.graph = graphviz.Digraph('UML_Diagram',
                                      graph_attr={'rankdir':'TB', 'size':"8,11", 'ratio':'compress'})
        self.files_content = {}
        # Adopted new collection names as per user's snippet
        self.class_info = {}
        self.function_info = {}
        self.api_routes = {}
        self.direct_references = set()
        self.indirect_references = set()
    
    def analyze_folder(self, max_depth=3):
        exclude_dirs = {'.git', '.github', '__pycache__', 'venv', '.venv', 'node_modules'}
        code_extensions = {'.py', '.js', '.java', '.c', '.cpp', '.go', '.rs', '.ts'}
        
        for root, _, files in os.walk(self.root_path):
            # Skip excluded directories
            if any(ex_dir in Path(root).parts for ex_dir in exclude_dirs):
                continue
                
            for file in files:
                file_path = Path(root) / file
                # Only process code files
                if file_path.suffix in code_extensions:
                    self.analyze_file(file_path)

    def analyze_file(self, file_path):
        """Analyze a single file's structure and dependencies"""
        try:
            content = read_file_with_fallback(file_path)
            self.files_content[file_path] = content
            language = detect_language(str(file_path), content)
            
            # Re-enabled structural analysis as per user's snippet
            self._extract_classes_and_functions(file_path, content, language)
            self._extract_api_endpoints(file_path, content, language)
            self._extract_dependencies(file_path, content, language)

        except Exception as e:
            print(f"Error analyzing {file_path.name}: {str(e)}")
            traceback.print_exc()

    def _extract_classes_and_functions(self, file_path, content, language):
        """Extract class and function info for UML labels, using AST for Python and Tree-sitter for others."""
        classes = {}
        functions = {}

        if language == 'python':
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Extract methods within classes
                        classes[node.name] = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    elif isinstance(node, ast.FunctionDef):
                        # Top-level functions
                        functions[node.name] = []
                self.class_info[file_path] = classes
                self.function_info[file_path] = functions
                return
            except Exception as e:
                print(f"AST parsing failed for {file_path.name}, falling back to Tree-sitter/Regex: {str(e)}")
        
        # Fallback to Tree-sitter for other languages and Python if AST fails
        parser_info = PARSERS.get(language)
        if parser_info:
            parser, lang = parser_info
            query_str = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG['default'])['tree_sitter_query']
            
            if query_str:
                try:
                    tree = parser.parse(content.encode('utf8'))
                    query = lang.query(query_str)
                    captures = query.captures(tree.root_node)

                    temp_classes = []
                    temp_functions = []
                    for node, capture_name in captures:
                        if capture_name == 'name': # Assuming 'name' capture directly gives class/func name
                            if node.parent.type.startswith('class') or node.parent.type.endswith('class_declaration') or node.parent.type.endswith('class_specifier'):
                                temp_classes.append(node.text.decode('utf8'))
                            elif node.parent.type.startswith('function') or node.parent.type.endswith('method_declaration'):
                                temp_functions.append(node.text.decode('utf8'))
                    
                    # Store as dictionaries if needed, or simply lists as per user's snippet's usage
                    self.class_info[file_path] = {cls_name: [] for cls_name in temp_classes} # Simplified
                    self.function_info[file_path] = {func_name: [] for func_name in temp_functions} # Simplified
                    return
                except Exception as e:
                    print(f"Tree-sitter parsing failed for {file_path.name}: {str(e)}")

        # Final regex fallback for all languages if AST/Tree-sitter fails
        print(f"Using regex fallback for {file_path.name}")
        class_regex = r'(?:class|struct|interface|enum)\s+(\w+)'
        func_regex = r'(?:func|def|function)\s+(\w+)'
        
        self.class_info[file_path] = {name: [] for name in re.findall(class_regex, content)}
        self.function_info[file_path] = {name: [] for name in re.findall(func_regex, content)}


    def _extract_dependencies(self, file_path, content, language):
        """Extract imports/includes and resolve them to internal file paths."""
        patterns = {
            'python': r'^(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))',
            'java': r'^import\s+([\w\.]+);',
            'javascript': r'(?:from|require\([\'"])([\w\.\/]+)',
            'c': r'#include\s+[<"]([\w\.\/]+)[>"]',
            'default': r'(?:import|include)\s+([\w\.\/]+)' # Generic fallback
        }
        
        pattern = patterns.get(language, patterns['default'])
        raw_imports = re.findall(pattern, content, re.MULTILINE)
        
        # Flatten list of tuples for python pattern
        if language == 'python':
            imports = [item for sublist in raw_imports for item in sublist if item]
        else:
            imports = raw_imports

        for imp in imports:
            resolved_target_file_path = None
            
            # Attempt to resolve import/module name to an actual file path
            # Prioritize exact matches, then stem matches, then path component matches
            target_module_as_path = imp.replace('.', os.sep) # Convert module.name to module/name
            
            # Check for direct file or folder existence
            potential_paths = [
                self.root_path / f"{target_module_as_path}.py", # .py extension
                self.root_path / f"{target_module_as_path}.js", # .js extension
                self.root_path / f"{target_module_as_path}.java", # .java extension
                self.root_path / f"{target_module_as_path}",    # folder (for __init__.py)
                self.root_path / f"{target_module_as_path}/__init__.py" # For python packages
            ]
            
            # Add other common extensions dynamically based on project files
            for known_file_path in self.files_content.keys():
                known_suffix = Path(known_file_path).suffix
                if known_suffix and known_suffix != '.': # Ensure it's a valid suffix
                    potential_paths.append(self.root_path / f"{target_module_as_path}{known_suffix}")

            for p_path in potential_paths:
                if p_path.exists() and p_path.is_file():
                    resolved_target_file_path = p_path.resolve()
                    break
                elif p_path.is_dir(): # If it's a directory, assume it's a package
                    resolved_target_file_path = p_path.resolve()
                    break

            # If not resolved by direct path, check if any file stem matches part of the import
            if not resolved_target_file_path:
                for existing_file_path in self.files_content.keys():
                    # Check if the stem of an existing file is in the import string
                    # e.g., 'utils' in 'my_project.utils.helpers'
                    if Path(existing_file_path).stem in imp.split('.')[-1]: # Check last component
                        resolved_target_file_path = Path(existing_file_path).resolve()
                        break
            
            if resolved_target_file_path and resolved_target_file_path != file_path.resolve():
                # Ensure the resolved path is actually one of the analyzed files
                if resolved_target_file_path in self.files_content:
                    self.direct_references.add((file_path, resolved_target_file_path))
                else:
                    # If it's an external file found, but not in our analysis scope, add as indirect
                    self.indirect_references.add((file_path, resolved_target_file_path, "External Dependency"))


    def _extract_api_endpoints(self, file_path, content, language):
        """Extract API endpoints based on language patterns, populating self.api_routes."""
        patterns = {
            'python': r'@app\.route\([\'"]([^\'"]+)',
            'java': r'@(?:GetMapping|PostMapping|RequestMapping)\([^)]*value\s*=\s*{?"([^"}]+)',
            'javascript': r'\.(?:get|post)\([\'"]([^\'"]+)',
            'default': r'(?:GET|POST)\s+([^\s]+)' # Generic fallback
        }
        
        pattern = patterns.get(language, patterns['default'])
        self.api_routes[file_path] = re.findall(pattern, content)
        
    def create_diagram(self, current_file_path=None):
        with self.graph.subgraph(name='cluster_0') as root_cluster:
            root_cluster.attr(label=str(self.root_path.name))
 
            for file_path in self.files_content.keys():
                node_name = str(file_path.relative_to(self.root_path))
                label = [node_name]
 
                if file_path in self.class_info:
                    label.append("\nClasses:")
                    for class_name, methods in self.class_info[file_path].items():
                        label.append(f"  {class_name}")
                        for method in methods:
                            label.append(f"    - {method}")
 
                if file_path in self.function_info:
                    label.append("\nFunctions:")
                    for func_name in self.function_info[file_path]:
                        label.append(f"  - {func_name}")
 
                if file_path in self.api_routes:
                    label.append("\nAPI Routes:")
                    for route in self.api_routes[file_path]:
                        label.append(f"  {route}")
 
                is_current_file = current_file_path and Path(current_file_path).resolve() == file_path.resolve()
 
                node_attrs = {
                    'shape': 'box',
                    'style': 'filled',
                    'fillcolor': 'lightgreen' if is_current_file else 'white'
                }
 
                root_cluster.node(node_name, '\n'.join(label), **node_attrs)
 
        for source, target in self.direct_references:
            source_name = str(Path(source).relative_to(self.root_path))
            target_path = str(target).replace('.', '/')
            for file_path in self.files_content.keys():
                if Path(file_path).stem in target_path:
                    target_name = str(Path(file_path).relative_to(self.root_path))
                    self.graph.edge(source_name, target_name, color='blue')
 
        for source, target, ref_type in self.indirect_references:
            source_name = str(Path(source).relative_to(self.root_path))
            target_name = str(Path(target).relative_to(self.root_path))
            self.graph.edge(source_name, target_name,
                            style='dotted',
                            color='red',
                            label=ref_type)

    # def create_diagram(self, current_file_path=None):
    #     with self.graph.subgraph(name='cluster_0') as root_cluster:
    #         root_cluster.attr(label=str(self.root_path.name))
    #         for file_path in self.files_content:
    #             node_name = str(file_path.relative_to(self.root_path))
    #             label = [node_name]
                
    #             if file_path in self.class_info:
    #                 label.append("\nClasses:")
    #                 label.extend(f"  {cls}" for cls in self.class_info[file_path])
                
    #             if file_path in self.function_info:
    #                 label.append("\nFunctions:")
    #                 label.extend(f"  - {fn}" for fn in self.function_info[file_path])
                
    #             if file_path in self.api_routes:
    #                 label.append("\nAPI Routes:")
    #                 label.extend(f"  {route}" for route in self.api_routes[file_path])

    #             is_current = current_file_path and Path(current_file_path).resolve() == file_path.resolve()
    #             root_cluster.node(node_name, '\n'.join(label),
    #                              shape='box', style='filled',
    #                              fillcolor='lightgreen' if is_current else 'white')

    #     for source, target in self.direct_references:
    #         source_name = str(Path(source).relative_to(self.root_path))
    #         target_path = target.replace('.', '/')
    #         for file_path in self.files_content:
    #             if Path(file_path).stem in target_path:
    #                 self.graph.edge(source_name, 
    #                               str(Path(file_path).relative_to(self.root_path)),
    #                               color='blue')

    #     for source, target, ref_type in self.indirect_references:
    #         self.graph.edge(str(Path(source).relative_to(self.root_path)),
    #                       str(Path(target).relative_to(self.root_path)),
    #                       style='dotted', color='red', label=ref_type)


    def save_diagram(self, filename):
        """Save the diagram to a PNG file"""
        try:
            self.graph.render(filename, format='png', cleanup=True)
            print(f"Diagram saved as {filename}.png")
        except Exception as e:
            print(f"Failed to save diagram: {str(e)}")
            traceback.print_exc()
def add_uml_diagram_to_workflow(state: FileState) -> FileState:
    """Creates UML diagram and adds it to the report state"""
    analyzer = FileAnalyzer(state['dir_path'])
    analyzer.analyze_folder()
    
    output_file = f"uml_{Path(state['file_path']).stem}"
    
    # Pass the current file path when creating the diagram
    analyzer.create_diagram(current_file_path=state['file_path'])
    analyzer.save_diagram(output_file)
    
    diagram_path = f"{output_file}.png"
    if os.path.exists(diagram_path):
        state['uml_diagram'] = diagram_path
    else:
        print(f"Warning: UML diagram file not found at {diagram_path}")
    return state

def create_workflow(include_uml=True):
    workflow = StateGraph(FileState)
    
    # # Add nodes to the workflow
    # workflow.add_node("analyze_full_code", analyze_full_code)
    # workflow.add_node("extract_blocks", extract_code_blocks)
    # workflow.add_node("generate_screenshots", generate_code_block_screenshot)
    # workflow.add_node("generate_pdf", generate_pdf_report)
    
    # if include_uml:
    #     workflow.add_node("create_uml", add_uml_diagram_to_workflow)

    # # Build the workflow edges
    # workflow.add_edge(START, "analyze_full_code")
    # workflow.add_edge("analyze_full_code", "extract_blocks")
    # workflow.add_edge("extract_blocks", "generate_screenshots")

    # if include_uml:
    #     workflow.add_edge("generate_screenshots", "create_uml")
    #     workflow.add_edge("create_uml", "generate_pdf")
    # else:
    #     workflow.add_edge("generate_screenshots", "generate_pdf")

    # workflow.add_edge("generate_pdf", END)
        # Add nodes
    workflow.add_node("analyze_full_code", analyze_full_code)
    workflow.add_node("extract_blocks", extract_code_blocks)
    workflow.add_node("generate_screenshots", generate_code_block_screenshot)
    workflow.add_node("create_uml", add_uml_diagram_to_workflow)
    workflow.add_node("generate_pdf", generate_pdf_report)
   
    # Add edges
    workflow.add_edge(START, "analyze_full_code")
    workflow.add_edge("analyze_full_code", "extract_blocks")
    workflow.add_edge("extract_blocks", "generate_screenshots")
    workflow.add_edge("generate_screenshots", "create_uml")  
    workflow.add_edge("create_uml", "generate_pdf")          
    workflow.add_edge("generate_pdf", END)
    
    return workflow.compile()

def process_directory(directory_path: str, include_uml=True):
    workflow = create_workflow(include_uml)
    directory_path = str(Path(directory_path).resolve())
    reports_dir = Path("code_analysis_reports")
    reports_dir.mkdir(exist_ok=True)

    exclude_dirs = {'venv', '__pycache__', '.venv', 'env', 'node_modules', 'target', 'build', '.git', '.github'}
    extensions = {
        '.py', '.js', '.java', '.c', '.cpp', '.h', '.hpp',
        '.go', '.rs', '.ts', '.php', '.rb', '.swift', '.kt', '.cs'
    }

    all_files = [
        f for f in Path(directory_path).rglob('*')
        if f.is_file()
        and f.suffix.lower() in extensions
        and not any(part in exclude_dirs for part in f.parts)
    ]

    print(f"Found {len(all_files)} files to process")

    for i, file_path in enumerate(all_files, 1):
        print(f"Processing {i}/{len(all_files)}: {file_path.name}")

        initial_state = FileState(
            file_path=str(file_path),
            content="",
            language="",
            code_blocks=[],
            summaries={},
            screenshots={},
            pdf_path=str(reports_dir / f"report_{file_path.stem}.pdf"),
            dir_path=directory_path,
            uml_diagram=""
        )

        try:
            workflow.invoke(initial_state)
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    input_path = "https://github.com/any_repo/" # works with both actual git repo or local path(use r for local path)
    process_input_path(input_path)
