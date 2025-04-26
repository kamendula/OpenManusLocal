# main.py 与 run_flow.py 的区别说明

## 1. 基本功能对比

### main.py
- 简单对话模式
- 直接使用 Manus 智能体处理用户输入
- 适合基础的对话场景
- 处理方式简单直接

### run_flow.py
- 流程处理模式
- 使用 FlowFactory 系统处理用户输入
- 支持不同类型的处理流程（如 Planning 流程）
- 适合需要复杂处理的场景

## 2. 技术实现差异

### main.py
```python
# 主要特点：
- 直接实例化 Manus 智能体
- 简单的输入处理流程
- 基础的错误处理机制
- 程序结束时会自动清理资源
```

### run_flow.py
```python
# 主要特点：
- 使用 FlowFactory 创建处理流程
- 支持超时处理（默认1小时）
- 更完善的错误处理机制
- 支持不同类型的流程处理
```

## 3. 使用场景建议

### 使用 main.py 的场景
- 简单的对话交互
- 不需要复杂处理流程的任务
- 快速测试和原型开发
- 资源受限的环境

### 使用 run_flow.py 的场景
- 需要多步骤处理的任务
- 需要计划执行的任务
- 需要长时间运行的任务
- 需要更完善的错误处理

## 4. 性能对比

### main.py
- 启动速度快
- 资源占用少
- 响应时间短
- 适合轻量级任务

### run_flow.py
- 启动时间较长
- 资源占用较多
- 处理时间可能较长
- 适合重量级任务

## 5. 错误处理能力

### main.py
- 基础错误处理
- 主要处理用户中断和空输入
- 简单的日志记录

### run_flow.py
- 完善的错误处理机制
- 支持超时处理
- 详细的错误日志记录
- 支持错误恢复机制

## 6. 使用建议

1. 如果是简单的对话任务，建议使用 `main.py`
2. 如果需要复杂的处理流程，建议使用 `run_flow.py`
3. 在资源受限的环境中，优先考虑使用 `main.py`
4. 在需要长时间运行的任务中，建议使用 `run_flow.py`

## 7. 示例代码

### main.py 示例
```python
async def main():
    agent = Manus()
    try:
        prompt = input("Enter your prompt: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return
        await agent.run(prompt)
    finally:
        await agent.cleanup()
```

### run_flow.py 示例
```python
async def run_flow():
    agents = {
        "manus": Manus(),
    }
    try:
        prompt = input("Enter your prompt: ")
        flow = FlowFactory.create_flow(
            flow_type=FlowType.PLANNING,
            agents=agents,
        )
        result = await asyncio.wait_for(
            flow.execute(prompt),
            timeout=3600,
        )
    except asyncio.TimeoutError:
        logger.error("Request processing timed out")
```

## 8. 总结

| 特性     | main.py  | run_flow.py |
| -------- | -------- | ----------- |
| 复杂度   | 简单     | 复杂        |
| 处理能力 | 基础     | 强大        |
| 资源占用 | 少       | 多          |
| 错误处理 | 基础     | 完善        |
| 适用场景 | 简单对话 | 复杂任务    |
| 启动速度 | 快       | 慢          |
| 扩展性   | 有限     | 强          |
