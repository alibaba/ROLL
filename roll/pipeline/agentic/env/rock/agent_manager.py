import asyncio
import concurrent.futures
import threading
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

from rock.sdk.sandbox.agent.base import Agent
from rock.sdk.sandbox.agent.config import AgentConfig
from rock.sdk.sandbox.agent.iflow_cli import IFlowCli, IFlowCliConfig
from rock.sdk.sandbox.agent.rock_agent import RockAgent, RockAgentConfig
from rock.sdk.sandbox.agent.swe_agent import SweAgent, SweAgentConfig
from rock.sdk.sandbox.client import Sandbox


class SingleLoopRunner:
    def __init__(self, name="single-loop"):
        self._ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        assert self._loop is not None
        return self._loop

    def call(self, coro, timeout: Optional[float] = None):
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result(timeout=timeout)

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def close(self, timeout: float = 5.0):
        if not self._loop:
            return

        async def _cleanup():
            current = asyncio.current_task()
            tasks = [t for t in asyncio.all_tasks() if t is not current]
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            await self._loop.shutdown_asyncgens()
            await self._loop.shutdown_default_executor()

        try:
            asyncio.run_coroutine_threadsafe(_cleanup(), self._loop).result(timeout=timeout)
        except Exception:
            pass

        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=timeout)

        try:
            self._loop.close()
        except Exception:
            pass


# ============ 工厂模式：创建 Agent ============
class AgentFactory:
    """Agent 工厂类"""

    _registry: Dict[str, tuple[Type[Agent], Type[AgentConfig]]] = {}

    @classmethod
    def register(
        cls,
        agent_type: str,
        agent_class: Type[Agent],
        config_class: Type[AgentConfig],
    ):
        """注册新的 agent 类型"""
        cls._registry[agent_type] = (agent_class, config_class)

    @classmethod
    def create(cls, agent_type: str, sandbox: Sandbox, config_dict: dict) -> tuple[Agent, AgentConfig]:
        """创建 agent 实例和对应的运行策略"""
        if agent_type not in cls._registry:
            raise ValueError(f"Unsupported agent type: {agent_type}")

        (
            agent_class,
            config_class,
        ) = cls._registry[agent_type]
        config = config_class(**config_dict)
        agent = agent_class(sandbox)
        return agent, config


AgentFactory.register("default", RockAgent, RockAgentConfig)
AgentFactory.register("swe-agent", SweAgent, SweAgentConfig)
AgentFactory.register("iflow-cli", IFlowCli, IFlowCliConfig)


# ============ 重构后的 AgentManager ============
class AgentManager:
    def __init__(self, sandbox: Sandbox, agent_config_dict: dict):
        self._sandbox: Sandbox = sandbox
        self._agent_config_dict: dict = agent_config_dict

        agent_type = agent_config_dict.get("agent_type")
        if not agent_type:
            raise ValueError("agent_type is required in config")

        # 使用工厂创建 agent 和运行策略
        agent, agent_config = AgentFactory.create(agent_type, self._sandbox, agent_config_dict)
        self._sandbox.agent = agent
        self._agent_config = agent_config
        self._loop_runner = SingleLoopRunner()
        self._agent_run_future = None

    @property
    def agent(self) -> Agent:
        return self._sandbox.agent

    def install_agent(self) -> None:
        self._loop_runner.call(self.agent.install(self._agent_config))

    def start_agent(self, prompt: str):
        fut = self._loop_runner.submit(self.agent.run(prompt=prompt))
        self._agent_run_future = fut
        return fut

    def anti_call_llm(
        self,
        index: int,
        response_payload: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> str:
        return self._loop_runner.call(self.agent.model_service.anti_call_llm(index, response_payload), timeout=timeout)

    def close(self, timeout: float = 10.0, cancel_agent: bool = True):
        fut = self._agent_run_future
        if fut:
            if not fut.done():
                try:
                    fut.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    if cancel_agent:
                        fut.cancel()
                except Exception:
                    pass

        self._loop_runner.close(timeout=timeout)
