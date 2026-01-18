from prompt import manager_system_prompt
from tools.BasicTools import ask_user
from tools.ManagementTools import manager_tools, manager_parameter, task_manager, execute_task_with_worker
from BasicFunction import create_agent
import logger
import traceback
import time
import nest_asyncio

nest_asyncio.apply()


async def execute_task_with_manager(user_input: str, continue_from_previous: bool = False):
    """
    Execute complex, multi-step tasks using Manager Agent with intelligent task orchestration.
    
    Description:
        This function handles sophisticated tasks that require planning, coordination, and 
        multi-step execution. The Manager Agent analyzes the user's request, breaks it down
        into a structured task list (Todo List), and orchestrates the execution of each subtask 
        using Worker Agents. It provides comprehensive project management capabilities including 
        automatic retry on failure, progress tracking, and final report generation. Supports 
        iterative refinement based on user feedback.
    
    Parameters:
        user_input (str): 
            The user's request or requirement description. Should be comprehensive enough 
            for the Manager Agent to understand the full scope of work needed.
            For new tasks: Complete description of what needs to be accomplished
            For continued tasks: Additional requirements or feedback on previous results

        
        continue_from_previous (bool, optional): 
            Indicates whether this is a continuation of a previous task execution.
            - False (default): Start a new task with fresh task list
            - True: Continue from previous execution, preserving completed tasks and 
                    adding new tasks based on user feedback
    
    Returns:
        str: A comprehensive response to the user containing:
            - Direct answer to the user's original question
            - Key information extracted from task execution results
            - Summary of what was accomplished
            - Explanation of any failures (if applicable)
            
            The response is conversational and user-focused, avoiding technical 
            implementation details like "task completed" or "file created" unless 
            directly relevant to the user's question.
    """
    manager_agent = create_agent("deepseek-reasoner", manager_parameter, manager_tools, manager_system_prompt)

    if not continue_from_previous:
        logger.info("📌 当前步骤: 创建todo list")
        planning_prompt = f"""
        Please analyze the following user request and create a detailed task list (Todo List).
        User Request: {user_input}
        
        Use the create_todo_list tool to generate the task list. Tasks should be arranged in execution order, with dependencies taken into consideration.Each task description should be sufficiently detailed to enable the Worker Agent to understand and complete it.
    """
    else:
        logger.info("📌 当前步骤: 基于用户反馈调整任务")
        current_todo = task_manager.get_todo_list()
        planning_prompt = f"""
        The user has provided additional requirements or feedback on the previous results.
        
        Current Task List Status:
        {current_todo}
        
        User's New Requirements/Feedback: {user_input}
        
        Please create an updated task list that addresses the user's new requirements. You can:
        1. Add new tasks to handle the additional requirements
        2. Modify existing pending tasks if needed
        
        Use the create_todo_list tool to generate the updated task list.
    """

    result = await manager_agent.run(planning_prompt)
    logger.info(result.output)
    manager_history = list(result.all_messages())

    logger.info("")
    logger.info("=" * 60)
    logger.info("当前步骤: 生成最终报告")
    logger.info("=" * 60)

    final_summary = task_manager.get_final_summary()
    logger.info(final_summary)

    
    summary_prompt = f"""Task execution completed. Please respond directly to the user's original question based on the execution report below.

User's Original Question: {user_input}

Execution Report:
{final_summary}

Important Guidelines:
- Do not report task execution status (e.g., "file created", "task completed successfully")
- Respond directly to the user's question as if you were having a conversation
- Extract key information from the task results in the execution report to answer the user
- If task failures prevent a proper answer, briefly explain why the information could not be obtained

Examples:
- If the user asks "What's the weather like in Wenjiang?", respond with the weather conditions, not "Successfully queried the weather"
- If the user asks "Write me a script", tell them where the script was saved and what its main functions are
"""
    try:
        final_result = await manager_agent.run(summary_prompt)
        logger.info("")
        logger.info("=" * 60)
        logger.info("🎯 最终回复")
        logger.info("=" * 60)
        logger.info(final_result.output)
        return final_result.output
    except Exception as e:
        return final_summary


def run_agent_system(user_input: str, history: list = []):
    """
    任务协调系统入口，负责判断任务复杂度并调用相应的执行器
    
    Parameters:
        user_input: 用户输入的任务描述

        history: 对话历史
        
    Returns:
        list: 更新后的对话历史
    """
    system_prompt = """
    You are the task coordinating agent.
    1. Determine task complexity:
        - Simple tasks (single, explicit operation): Execute directly using `execute_task_with_worker`
        - Complex tasks (requiring multiple steps or planning): Execute using `execute_task_with_manager`
    2. After task execution, provide clear feedback on the results, then immediately end the current dialogue.
    3. Do not proactively ask the user if they are satisfied; the user will proactively inform you of their next requirements.
    4. If the tool call fails, clearly explain the reason for the failure to the user.
    5. Important: After executing a task using the tool, immediately summarize the results and end the dialogue, awaiting the user's next instruction.
"""

    agent = create_agent("deepseek-reasoner", manager_parameter,
                         [execute_task_with_manager, execute_task_with_worker, ask_user], system_prompt)

    start_time = time.time()
    
    result = agent.run_sync(user_input, message_history=history)
    
    elapsed = time.time() - start_time
    logger.info(f"[DEBUG] run_agent_system agent.run_sync() 完成，耗时 {elapsed:.2f} 秒")
    logger.info(result.output)
    history = list(result.all_messages())
    return history


def main():
    """主函数 - 交互式运行"""
    log = logger.get_logger()
    log.info("=" * 60)
    log.info("输入 '新任务' 可以清除上下文重新开始")
    log.info("输入 'quit' 或 'exit' 退出程序")
    log.info("=" * 60)

    is_first_input = True
    history = []

    while True:
        try:
            user_input = input("\n📝 请输入您的任务: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', '退出']:
                log.info("👋 再见！")
                break

            if '新任务' in user_input:
                task_manager.reset()
                history = []
                is_first_input = True
                continue

            if is_first_input:
                task_name = user_input[:30].replace(" ", "_")
                logger.setup_task_logger(task_name)
                is_first_input = False

            history = run_agent_system(user_input, history)

        except KeyboardInterrupt:
            log.info("\n\n👋 程序已中断，再见！")
            break
            
        except Exception as e:
            log.error(f"\n❌ 未预期的系统错误: {e}")
            log.error(f"详细信息:\n{traceback.format_exc()}")
            task_manager.reset()


if __name__ == "__main__":
    main()
