# 前言

OpenManus 是一个开源项目，由 MetaGPT 团队的几位成员在短短三个小时内构建完成。通过阅读其代码，我们可以深入了解 AI Agent 的框架设计与实现细节。对于希望构建自己 AI Agent 的开发者而言，这是一份极佳的学习资源。

# AI Agent 框架：以 OpenManus 为例

# 1. 整体架构

OpenManus 采用模块化架构设计，包含多个核心组件：

```
OpenManus
├── Agent (代理层)
│ ├── BaseAgent (基础抽象类)
│ ├── ReActAgent (思考-行动模式)
│ ├── ToolCallAgent (工具调用能力)
│ ├── PlanningAgent (规划能力)
│ ├── SWEAgent (软件工程能力)
│ └── Manus (通用代理)
├── LLM (语言模型层)
├── Memory (记忆层)
├── Tool (工具层)
│ ├── BaseTool (工具基类)
│ ├── PlanningTool (规划工具)
│ ├── PythonExecute (Python执行)
│ ├── GoogleSearch (搜索工具)
│ ├── BrowserUseTool (浏览器工具)
│ └── ... (其他工具)
├── Flow (工作流层)
│ ├── BaseFlow (基础流程)
│ └── PlanningFlow (规划流程)
└── Prompt (提示层)
```
这种模块化设计确保了高代码复用性、强扩展性和清晰的职责分离。

# 2. LLM 组件

LLM（大型语言模型）作为 Agent 的大脑，负责理解用户输入、生成响应和做出决策。OpenManus 通过 LLM 类封装了与语言模型的交互：

```python
class LLM:
    _instances: Dict[str, "LLM"] = {}  # 单例模式实现
    def __init__(
            self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
        ):
            if not hasattr(self, "client"):  # 仅初始化一次
                llm_config = llm_config or config.llm
                llm_config = llm_config.get(config_name, llm_config["default"])
                self.model = llm_config.model
                self.max_tokens = llm_config.max_tokens
                self.temperature = llm_config.temperature
                self.client = AsyncOpenAI(
                    api_key=llm_config.api_key, base_url=llm_config.base_url
                )
```

LLM 类提供两个核心方法：

- `ask`：发送一般对话请求
- `ask_tool`：发送带工具调用的请求

```python
async def ask_tool(
    self,
    messages: List[Union[dict, Message]],
    system_msgs: Optional[List[Union[dict, Message]]] = None,
    timeout: int = 60,
    tools: Optional[List[dict]] = None,
    tool_choice: Literal["none", "auto", "required"] = "auto",
    temperature: Optional[float] = None,
    **kwargs,
):
    # 格式化消息
    if system_msgs:
        system_msgs = self.format_messages(system_msgs)
        messages = system_msgs + self.format_messages(messages)
    else:
        messages = self.format_messages(messages)

    # 发送请求
    response = await self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        temperature=temperature or self.temperature,
        max_tokens=self.max_tokens,
        tools=tools,
        tool_choice=tool_choice,
        timeout=timeout,
        **kwargs,
    )
```

# 3. 记忆组件

记忆组件负责存储和管理 Agent 的对话历史，确保上下文的连贯性：

```python
class Memory(BaseModel):
    """存储和管理代理的对话历史"""

    messages: List[Message] = Field(default_factory=list)

    def add_message(self, message: Union[Message, dict]) -> None:
        """添加消息到记忆中"""
        if isinstance(message, dict):
            message = Message(**message)
        self.messages.append(message)

    def get_messages(self) -> List[Message]:
        """获取记忆中的所有消息"""
        return self.messages
```

记忆组件通过 `BaseAgent` 的 `update_memory` 方法与 Agent 紧密集成：

```python
def update_memory(
    self,
    role: Literal["user", "system", "assistant", "tool"],
    content: str,
    **kwargs,
) -> None:
    """向代理的记忆中添加消息"""
    message_map = {
        "user": Message.user_message,
        "system": Message.system_message,
        "assistant": Message.assistant_message,
        "tool": lambda content, **kw: Message.tool_message(content, **kw),
    }
    if role not in message_map:
        raise ValueError(f"不支持的消息角色: {role}")
    msg_factory = message_map[role]
    msg = msg_factory(content, **kwargs) if role == "tool" else msg_factory(content)
    self.memory.add_message(msg)
```

# 4. 工具组件

工具是 Agent 与外部世界交互的接口。OpenManus 基于 `BaseTool` 实现了灵活的工具系统：

```python
class BaseTool(ABC, BaseModel):
    name: str
    description: str
    parameters: Optional[dict] = None
    async def __call__(self, **kwargs) -> Any:
        """使用给定参数执行工具"""
        return await self.execute(**kwargs)
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """使用给定参数执行工具"""
    def to_param(self) -> Dict:
        """将工具转换为函数调用格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
```

工具执行的结果由 `ToolResult` 类表示：

```python
class ToolResult(BaseModel):
    """表示工具执行的结果"""
    output: Any = Field(default=None)
    error: Optional[str] = Field(default=None)
    system: Optional[str] = Field(default=None)
```

OpenManus 提供了多个内置工具，如 `PlanningTool`：

```python
class PlanningTool(BaseTool):
    """
    规划工具允许代理创建和管理解决复杂任务的计划。
    该工具提供创建计划、更新计划步骤和跟踪进度的功能
    """
    name: str = "planning"
    description: str = _PLANNING_TOOL_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "要执行的命令。可用命令：create, update, list, get, set_active, mark_step, delete",
                "enum": [
                    "create",
                    "update",
                    "list",
                    "get",
                    "set_active",
                    "mark_step",
                    "delete",
                ],
                "type": "string",
            },
            # 其他参数...
        },
        "required": ["command"],
    }
```

# 5. 规划组件

规划组件是 OpenManus 的核心特性之一，使 Agent 能够创建和管理计划，将复杂任务分解为可管理的步骤。规划组件包含两个主要部分：

1. `PlanningTool`：提供创建、更新和跟踪计划的功能
2. `PlanningAgent`：使用 `PlanningTool` 来规划和执行任务

```python
class PlanningAgent(ToolCallAgent):
    """
    创建和管理计划以解决任务的代理。
    该代理使用规划工具创建和管理结构化计划，
    并通过单个步骤跟踪进度直至任务完成。
    """
    name: str = "planning"
    description: str = "创建和管理计划以解决任务的代理"
    system_prompt: str = PLANNING_SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(PlanningTool(), Terminate())
    )

    # 步骤执行跟踪器
    step_execution_tracker: Dict[str, Dict] = Field(default_factory=dict)
    current_step_index: Optional[int] = None
```

`PlanningAgent` 的核心方法包括：

```python
async def think(self) -> bool:
    """根据计划状态决定下一步行动"""
    prompt = (
        f"当前计划状态:\n{await self.get_plan()}\n\n{self.next_step_prompt}"
        if self.active_plan_id
        else self.next_step_prompt
    )
    self.messages.append(Message.user_message(prompt))

    # 获取当前步骤索引
    self.current_step_index = await self._get_current_step_index()
    result = await super().think()
    # 将工具调用与当前步骤关联
    if result and self.tool_calls:
        # ...关联逻辑...
    return result
```

# 6. 流程组件

流程组件用于管理多个 Agent 的协作，以实现更复杂的任务处理工作流：

```python
class BaseFlow(BaseModel, ABC):
    """支持多个代理的执行流程基类"""

    agents: Dict[str, BaseAgent]
    tools: Optional[List] = None
    primary_agent_key: Optional[str] = None

    @property
    def primary_agent(self) -> Optional[BaseAgent]:
        """获取流程的主要代理"""
        return self.agents.get(self.primary_agent_key)

    @abstractmethod
    async def execute(self, input_text: str) -> str:
        """使用给定输入执行流程"""
```

`PlanningFlow` 是用于规划和执行任务的特定流程实现：

```python
class PlanningFlow(BaseFlow):
    """使用代理管理任务规划和执行的流程"""
    llm: LLM = Field(default_factory=lambda: LLM())
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)
    executor_keys: List[str] = Field(default_factory=list)
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")
    current_step_index: Optional[int] = None

    async def execute(self, input_text: str) -> str:
        """执行规划流程"""
        try:
            # 创建初始计划
            if input_text:
                await self._create_initial_plan(input_text)

            # 执行计划步骤
            while await self._has_next_step():
                # 获取当前步骤
                step_info = await self._get_current_step()

                # 选择合适的执行器
                executor = self.get_executor(step_info.get("type"))

                # 执行步骤
                result = await self._execute_step(executor, step_info)

                # 更新步骤状态
                await self._update_step_status(step_info["index"], "completed")

            # 完成计划
            return await self._finalize_plan()

        except Exception as e:
            # 处理异常
            return f"执行流程出错: {str(e)}"
```

# OpenManus 实现：代理的关键代码

OpenManus 的代理采用层次化架构设计，从基础代理构建到专门代理。这种设计确保了高代码复用性、强扩展性和清晰的职责分离。
