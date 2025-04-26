# OpenManus 架构文档

## 1. 入口点系统

入口点系统是 OpenManus 的启动入口，负责初始化和管理整个应用程序的运行。

### 1.1 main.py

`main.py` 是主要的启动入口，负责初始化 Manus 代理并处理用户输入。

```python
async def main():
    agent = Manus()
    try:
        prompt = input("Enter your prompt: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request...")
        await agent.run(prompt)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        await agent.cleanup()
```

主要功能：
- 创建 Manus 代理实例
- 接收用户输入
- 执行代理运行
- 处理异常情况
- 清理资源

### 1.2 run_mcp.py

`run_mcp.py` 是 MCP (Model Context Protocol) 服务器的启动入口。

```python
def run(self, transport: str = "stdio") -> None:
    self.register_all_tools()
    atexit.register(lambda: asyncio.run(self.cleanup()))
    logger.info(f"Starting OpenManus server ({transport} mode)")
    self.server.run(transport=transport)
```

主要功能：
- 注册所有工具
- 设置清理函数
- 启动 MCP 服务器
- 处理通信传输

### 1.3 run_flow.py

`run_flow.py` 是多代理流程的启动入口。

```python
async def execute(self, input_text: str) -> str:
    try:
        if not self.primary_agent:
            raise ValueError("No primary agent available")
        # ... 执行流程逻辑
    except Exception as e:
        logger.error(f"Error in PlanningFlow: {str(e)}")
        return f"Execution failed: {str(e)}"
```

主要功能：
- 初始化流程系统
- 执行多代理协作
- 处理执行结果
- 错误处理

## 2. 代理系统

代理系统是 OpenManus 的核心，负责实现智能体的基本行为和决策能力。

### 2.1 BaseAgent

`BaseAgent` 是所有代理的基类，定义了代理的基本属性和行为。

```python
class BaseAgent(BaseModel, ABC):
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")
    system_prompt: Optional[str] = Field(None, description="System-level instruction prompt")
    next_step_prompt: Optional[str] = Field(None, description="Prompt for determining next action")
    llm: LLM = Field(default_factory=LLM, description="Language model instance")
    memory: Memory = Field(default_factory=Memory, description="Agent's memory store")
    state: AgentState = Field(default=AgentState.IDLE, description="Current agent state")
```

主要功能：
- 定义代理的基本属性
- 管理代理状态
- 提供内存管理
- 处理系统提示

### 2.2 ReActAgent

`ReActAgent` 实现了思考-行动循环的代理模式。

```python
class ReActAgent(BaseAgent, ABC):
    @abstractmethod
    async def think(self) -> bool:
        """Process current state and decide next action"""

    @abstractmethod
    async def act(self) -> str:
        """Execute decided actions"""

    async def step(self) -> str:
        """Execute a single step: think and act."""
        should_act = await self.think()
        if not should_act:
            return "Thinking complete - no action needed"
        return await self.act()
```

主要功能：
- 实现思考-行动循环
- 提供抽象方法接口
- 管理执行步骤
- 处理决策流程

### 2.3 ToolCallAgent

`ToolCallAgent` 支持工具调用的代理实现。

```python
class ToolCallAgent(ReActAgent):
    available_tools: ToolCollection = ToolCollection(CreateChatCompletion(), Terminate())
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])
    tool_calls: List[ToolCall] = Field(default_factory=list)
```

主要功能：
- 管理可用工具集合
- 处理工具调用
- 支持特殊工具处理
- 维护工具调用状态

### 2.4 Manus

`Manus` 是主要的通用代理实现。

```python
class Manus(ToolCallAgent):
    name: str = "Manus"
    description: str = "A versatile agent that can solve various tasks using multiple tools"
    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(), BrowserUseTool(), StrReplaceEditor(), Terminate()
        )
    )
```

主要功能：
- 提供通用任务解决能力
- 集成多种工具
- 处理浏览器上下文
- 管理执行流程

## 3. 流程系统

流程系统负责管理多代理协作和任务执行流程。

### 3.1 BaseFlow

`BaseFlow` 是流程的基类，定义了流程的基本结构。

```python
class BaseFlow(BaseModel, ABC):
    agents: Dict[str, BaseAgent]
    tools: Optional[List] = None
    primary_agent_key: Optional[str] = None

    @property
    def primary_agent(self) -> Optional[BaseAgent]:
        return self.agents.get(self.primary_agent_key)

    @abstractmethod
    async def execute(self, input_text: str) -> str:
        """Execute the flow with given input"""
```

主要功能：
- 管理代理集合
- 定义主要代理
- 提供执行接口
- 支持工具集成

### 3.2 PlanningFlow

`PlanningFlow` 实现规划执行流程。

```python
class PlanningFlow(BaseFlow):
    llm: LLM = Field(default_factory=lambda: LLM())
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)
    executor_keys: List[str] = Field(default_factory=list)
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")
```

主要功能：
- 管理规划工具
- 执行计划步骤
- 跟踪执行状态
- 处理执行结果

### 3.3 FlowFactory

`FlowFactory` 负责创建不同类型的流程。

```python
class FlowFactory:
    @staticmethod
    def create_flow(
        flow_type: FlowType,
        agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]],
        **kwargs,
    ) -> BaseFlow:
        flows = {
            FlowType.PLANNING: PlanningFlow,
        }
        flow_class = flows.get(flow_type)
        if not flow_class:
            raise ValueError(f"Unknown flow type: {flow_type}")
        return flow_class(agents, **kwargs)
```

主要功能：
- 创建流程实例
- 管理流程类型
- 配置流程参数
- 提供错误处理

## 4. 工具系统

工具系统提供各种功能扩展，支持代理执行具体任务。

### 4.1 BaseTool

`BaseTool` 是所有工具的基类。

```python
class BaseTool:
    def __init__(self):
        self.name = self.__class__.__name__.lower()
        self.description = self.__doc__ or ""
        self.parameters = self._get_parameters()

    def to_param(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
```

主要功能：
- 定义工具接口
- 管理工具元数据
- 提供参数处理
- 支持序列化

### 4.2 ToolCollection

`ToolCollection` 管理工具集合。

```python
class ToolCollection:
    def __init__(self, *tools: BaseTool):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

    async def execute(self, *, name: str, tool_input: Dict[str, Any] = None) -> ToolResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid")
        try:
            result = await tool(**tool_input)
            return result
        except ToolError as e:
            return ToolFailure(error=e.message)
```

主要功能：
- 管理工具集合
- 提供工具查找
- 执行工具调用
- 处理执行结果

### 4.3 具体工具实现

#### PythonExecute
```python
class PythonExecute(BaseTool):
    """Execute Python code in a sandbox environment."""
    async def execute(self, code: str) -> str:
        return await SANDBOX_CLIENT.run_command(f"python -c '{code}'")
```

#### BrowserUseTool
```python
class BrowserUseTool(BaseTool):
    """Interact with web browsers."""
    async def execute(self, url: str, action: str) -> str:
        return await self.browser_context_helper.execute_action(url, action)
```

#### StrReplaceEditor
```python
class StrReplaceEditor(BaseTool):
    """Edit text content with string replacements."""
    async def execute(self, content: str, replacements: Dict[str, str]) -> str:
        for old, new in replacements.items():
            content = content.replace(old, new)
        return content
```

#### Terminate
```python
class Terminate(BaseTool):
    """Terminate the current execution."""
    async def execute(self) -> str:
        return "Execution terminated"
```

#### PlanningTool
```python
class PlanningTool(BaseTool):
    """Manage task planning and execution."""
    async def execute(self, command: str, plan_id: str, **kwargs) -> str:
        if command == "create":
            return await self._create_plan(plan_id, **kwargs)
        elif command == "mark_step":
            return await self._mark_step(plan_id, **kwargs)
```

## 5. 沙箱系统

沙箱系统提供安全的执行环境，用于运行不受信任的代码。

### 5.1 BaseSandboxClient

`BaseSandboxClient` 定义沙箱客户端接口。

```python
class BaseSandboxClient(ABC):
    @abstractmethod
    async def create(self, config: Optional[SandboxSettings] = None) -> None:
        """Creates sandbox."""

    @abstractmethod
    async def run_command(self, command: str, timeout: Optional[int] = None) -> str:
        """Executes command."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleans up resources."""
```

主要功能：
- 定义沙箱接口
- 提供创建方法
- 支持命令执行
- 管理资源清理

### 5.2 LocalSandboxClient

`LocalSandboxClient` 实现本地沙箱客户端。

```python
class LocalSandboxClient(BaseSandboxClient):
    def __init__(self):
        self.sandbox: Optional[DockerSandbox] = None

    async def create(self, config: Optional[SandboxSettings] = None) -> None:
        self.sandbox = DockerSandbox(config)
        await self.sandbox.create()
```

主要功能：
- 管理 Docker 沙箱
- 处理命令执行
- 提供文件操作
- 管理资源生命周期

### 5.3 DockerSandbox

`DockerSandbox` 实现 Docker 容器沙箱。

```python
class DockerSandbox:
    def __init__(self, config: Optional[SandboxSettings] = None):
        self.config = config or SandboxSettings()
        self.client = docker.from_env()
        self.container: Optional[Container] = None
        self.terminal: Optional[AsyncDockerizedTerminal] = None

    async def create(self) -> "DockerSandbox":
        host_config = self.client.api.create_host_config(
            mem_limit=self.config.memory_limit,
            cpu_period=100000,
            cpu_quota=int(100000 * self.config.cpu_limit),
            network_mode="none" if not self.config.network_enabled else "bridge",
        )
        # ... 创建和配置容器
```

主要功能：
- 管理 Docker 容器
- 配置资源限制
- 提供执行环境
- 处理文件操作

### 5.4 SandboxManager

`SandboxManager` 管理多个沙箱实例。

```python
class SandboxManager:
    def __init__(self, max_sandboxes: int = 100, idle_timeout: int = 3600):
        self.max_sandboxes = max_sandboxes
        self.idle_timeout = idle_timeout
        self._sandboxes: Dict[str, DockerSandbox] = {}
        self._last_used: Dict[str, float] = {}
```

主要功能：
- 管理沙箱池
- 处理资源分配
- 监控使用状态
- 自动清理资源

### 5.5 AsyncDockerizedTerminal

`AsyncDockerizedTerminal` 提供异步 Docker 终端功能。

```python
class AsyncDockerizedTerminal:
    def __init__(self, container: Union[str, Container], working_dir: str = "/workspace"):
        self.container = container
        self.working_dir = working_dir
        self.session: Optional[DockerSession] = None

    async def run_command(self, cmd: str, timeout: Optional[int] = None) -> str:
        if not self.session:
            raise RuntimeError("Terminal not initialized")
        return await self.session.execute(cmd, timeout)
```

主要功能：
- 提供终端接口
- 执行命令
- 处理超时
- 管理会话

## 6. LLM 系统

LLM 系统负责与语言模型的交互和令牌管理。

### 6.1 LLM

`LLM` 类管理语言模型交互。

```python
class LLM:
    def __init__(self, config_name: str = "default", llm_config: Optional[LLMSettings] = None):
        llm_config = llm_config or config.llm
        self.model = llm_config.model
        self.max_tokens = llm_config.max_tokens
        self.temperature = llm_config.temperature
        self.api_type = llm_config.api_type
        self.api_key = llm_config.api_key
        self.base_url = llm_config.base_url

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    async def ask(self, messages: List[Union[dict, Message]], system_msgs: Optional[List[Union[dict, Message]]] = None) -> str:
        # ... 实现语言模型调用
```

主要功能：
- 管理模型配置
- 处理 API 调用
- 实现重试机制
- 处理令牌限制

### 6.2 TokenCounter

`TokenCounter` 负责令牌计数和管理。

```python
class TokenCounter:
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    def count_message_tokens(self, messages: List[dict]) -> int:
        total_tokens = self.FORMAT_TOKENS
        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS
            tokens += self.count_text(message.get("role", ""))
            tokens += self.count_content(message.get("content", ""))
            total_tokens += tokens
        return total_tokens
```

主要功能：
- 计算消息令牌
- 处理图像令牌
- 管理令牌限制
- 提供优化建议

## 7. 配置系统

配置系统管理应用程序的各种设置和参数。

### 7.1 Config

`Config` 类管理全局配置。

```python
class Config:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config = None
                    self._load_initial_config()
                    self._initialized = True
```

主要功能：
- 实现单例模式
- 管理配置加载
- 提供线程安全
- 处理配置更新

### 7.2 AppConfig

`AppConfig` 定义应用程序配置结构。

```python
class AppConfig(BaseModel):
    llm: Dict[str, LLMSettings]
    sandbox: Optional[SandboxSettings] = None
    browser_config: Optional[BrowserSettings] = None
    search_config: Optional[SearchSettings] = None
    mcp_config: Optional[MCPSettings] = None
```

主要功能：
- 定义配置结构
- 管理配置验证
- 提供类型安全
- 支持配置继承

### 7.3 各种设置类

#### LLMSettings
```python
class LLMSettings(BaseModel):
    model: str = Field(..., description="Model name")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key")
    max_tokens: int = Field(4096, description="Maximum number of tokens per request")
```

#### SandboxSettings
```python
class SandboxSettings(BaseModel):
    use_sandbox: bool = Field(False, description="Whether to use the sandbox")
    image: str = Field("python:3.12-slim", description="Base image")
    work_dir: str = Field("/workspace", description="Container working directory")
```

#### BrowserSettings
```python
class BrowserSettings(BaseModel):
    headless: bool = Field(False, description="Whether to run browser in headless mode")
    disable_security: bool = Field(True, description="Disable browser security features")
```

#### SearchSettings
```python
class SearchSettings(BaseModel):
    engine: str = Field(default="Google", description="Search engine the llm to use")
    fallback_engines: List[str] = Field(default_factory=lambda: ["DuckDuckGo", "Baidu", "Bing"])
```

#### MCPSettings
```python
class MCPSettings(BaseModel):
    server_reference: str = Field("app.mcp.server", description="Module reference for the MCP server")
```

## 8. 数据模型

数据模型定义了系统使用的核心数据结构。

### 8.1 Message

`Message` 类定义消息结构。

```python
class Message(BaseModel):
    role: ROLE_TYPE = Field(...)
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(default=None)

    @classmethod
    def user_message(cls, content: str, base64_image: Optional[str] = None) -> "Message":
        return cls(role=Role.USER, content=content, base64_image=base64_image)
```

主要功能：
- 定义消息结构
- 提供消息工厂方法
- 支持工具调用
- 处理图像内容

### 8.2 Memory

`Memory` 类管理代理的记忆存储。

```python
class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
```

主要功能：
- 管理消息历史
- 控制内存大小
- 提供消息操作
- 支持序列化

### 8.3 ToolCall

`ToolCall` 类定义工具调用结构。

```python
class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Function
```

主要功能：
- 定义工具调用格式
- 管理调用标识
- 处理函数参数
- 支持序列化

### 8.4 Function

`Function` 类定义函数结构。

```python
class Function(BaseModel):
    name: str
    arguments: str
```

主要功能：
- 定义函数格式
- 管理函数名称
- 处理函数参数
- 支持序列化
