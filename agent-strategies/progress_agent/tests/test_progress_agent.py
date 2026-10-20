import json
import unittest
from copy import deepcopy
from unittest.mock import Mock

from dify_plugin.entities.model import ModelFeature
from dify_plugin.entities.model.llm import LLMResult, LLMResultChunk, LLMResultChunkDelta, LLMUsage
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    SystemPromptMessage,
    UserPromptMessage,
)
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.interfaces.agent import AgentModelConfig, ToolEntity

from strategies.function_calling import FunctionCallingAgentStrategy


def tool_call(round_number, name="lookup", suffix=""):
    return AssistantPromptMessage.ToolCall(
        id=f"call-{round_number}{suffix}",
        type="function",
        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
            name=name, arguments='{"q":"test"}'
        ),
    )


def result(content="", calls=None):
    return LLMResult(
        model="test",
        message=AssistantPromptMessage(content=content, tool_calls=calls or []),
        usage=LLMUsage.empty_usage(),
    )


class TestProgressAgent(unittest.TestCase):
    def run_agent(self, responses, *, params=None, streaming=False):
        session = Mock()
        prompts = []
        tools = []

        def invoke(**kwargs):
            prompts.append(deepcopy(kwargs["prompt_messages"]))
            tools.append(deepcopy(kwargs.get("tools")))
            value = responses[len(prompts) - 1]
            if not streaming:
                return value

            def chunks():
                yield LLMResultChunk(
                    model="test",
                    delta=LLMResultChunkDelta(index=0, message=value.message, usage=value.usage),
                )

            return chunks()

        session.model.llm.invoke.side_effect = invoke
        session.tool.invoke.side_effect = lambda **kwargs: iter(
            [
                ToolInvokeMessage(
                    type=ToolInvokeMessage.MessageType.TEXT,
                    message=ToolInvokeMessage.TextMessage(text="observation"),
                ),
            ]
        )
        model = AgentModelConfig(provider="test", model="test", mode="chat")
        if streaming:
            model.entity = Mock(features=[ModelFeature.STREAM_TOOL_CALL])
            model.completion_params = {}
        parameters = {
            "model": model,
            "instruction": "Help with the task",
            "query": "Find the answer",
            "tools": [
                ToolEntity.model_validate(
                    {
                        "identity": {
                            "author": "test",
                            "name": "lookup",
                            "label": {"en_US": "lookup"},
                            "provider": "test",
                        },
                        "provider_type": "workflow",
                        "runtime_parameters": {},
                    }
                )
            ],
        }
        parameters.update(params or {})
        strategy = FunctionCallingAgentStrategy(runtime=Mock(), session=session)
        output = list(strategy._invoke(parameters))
        return output, prompts, tools, session

    def text(self, output):
        return "".join(
            m.message.text for m in output if m.type == ToolInvokeMessage.MessageType.TEXT
        )

    def metadata(self, output):
        return next(
            m.message.json_object["iteration_metadata"]
            for m in output
            if m.type == ToolInvokeMessage.MessageType.JSON
            and "iteration_metadata" in m.message.json_object
        )

    def budget(self, messages):
        system = next(m for m in messages if isinstance(m, SystemPromptMessage))
        start = system.content.index("<iteration_budget>") + len("<iteration_budget>")
        end = system.content.index("</iteration_budget>", start)
        return json.loads(system.content[start:end])

    def test_only_runtime_state_is_added_to_user_configured_instruction(self):
        instruction = "用户自定义规则：检查点先评估进展；收尾时说明未完成事项。"
        responses = [result(calls=[tool_call(i)]) for i in range(1, 30)] + [result("Done")]
        _, prompts, _, _ = self.run_agent(responses, params={"instruction": instruction})
        for messages in prompts:
            system = next(m for m in messages if isinstance(m, SystemPromptMessage))
            state = self.budget(messages)
            expected = f"{instruction}\n\n<iteration_budget>{json.dumps(state)}</iteration_budget>"
            self.assertEqual(system.content, expected)

    def test_default_thirty_round_budget_and_checkpoints(self):
        responses = [result(calls=[tool_call(i)]) for i in range(1, 30)] + [result("Summary")]
        output, prompts, tools, session = self.run_agent(responses)
        self.assertEqual(len(prompts), 30)
        self.assertEqual(session.tool.invoke.call_count, 29)
        self.assertEqual(tools[-1], [])
        for i, messages in enumerate(prompts, start=1):
            state = self.budget(messages)
            self.assertEqual(state["current_iteration"], i)
            self.assertEqual(state["is_checkpoint"], i in (20, 25))
            self.assertEqual(sum(str(m.content).count("<iteration_budget>") for m in messages), 1)
        self.assertEqual(self.metadata(output)["checkpoints_reached"], [20, 25])
        self.assertTrue(self.metadata(output)["finalization_reached"])
        self.assertIn("Summary", self.text(output))

    def test_runtime_string_settings_change_checkpoints(self):
        responses = [result(calls=[tool_call(i)]) for i in range(1, 5)] + [result("Done")]
        output, prompts, tools, session = self.run_agent(
            responses,
            params={
                "maximum_iterations": "5",
                "progress_check_start": "2",
                "progress_check_interval": "2",
                "final_response_rounds": "1",
            },
        )
        self.assertEqual(self.metadata(output)["checkpoints_reached"], [2, 4])
        self.assertEqual(session.tool.invoke.call_count, 4)
        self.assertEqual(tools[-1], [])

    def test_model_can_ask_user_and_end_early(self):
        output, prompts, _, session = self.run_agent([result("Which account should I use?")])
        self.assertEqual(len(prompts), 1)
        session.tool.invoke.assert_not_called()
        self.assertIn("Which account", self.text(output))
        self.assertFalse(self.metadata(output)["finalization_reached"])

    def test_reserved_phase_blocks_unexpected_calls(self):
        output, prompts, tools, session = self.run_agent(
            [result(calls=[tool_call(1)]), result(calls=[tool_call(2)])],
            params={"maximum_iterations": 3, "final_response_rounds": 2},
        )
        self.assertEqual(len(prompts), 2)
        self.assertEqual(session.tool.invoke.call_count, 1)
        self.assertEqual(tools[-1], [])
        self.assertEqual(self.metadata(output)["blocked_tool_calls"], 1)
        self.assertTrue(self.text(output).strip())

    def test_one_round_budget_never_executes_tools(self):
        output, prompts, tools, session = self.run_agent(
            [result(calls=[tool_call(1)])],
            params={"maximum_iterations": 1},
        )
        session.tool.invoke.assert_not_called()
        self.assertEqual(len(prompts), 1)
        self.assertEqual(tools, [[]])
        self.assertTrue(self.text(output).strip())

    def test_malformed_final_phase_tool_arguments_are_never_executed(self):
        call = tool_call(1)
        call.function.arguments = '{"q":'
        output, _, _, session = self.run_agent(
            [result(calls=[call])],
            params={"maximum_iterations": 1},
        )
        session.tool.invoke.assert_not_called()
        self.assertEqual(self.metadata(output)["blocked_tool_calls"], 1)

    def test_malformed_final_phase_streaming_call_is_never_executed(self):
        call = tool_call(1)
        call.function.arguments = "not-json"
        output, _, _, session = self.run_agent(
            [result(calls=[call])],
            params={"maximum_iterations": 1},
            streaming=True,
        )
        session.tool.invoke.assert_not_called()
        self.assertEqual(self.metadata(output)["stop_reason"], "tool_call_blocked")

    def test_empty_final_answer_has_fallback(self):
        output, _, _, session = self.run_agent([result("  ")], params={"maximum_iterations": 1})
        self.assertTrue(self.text(output).strip())
        session.tool.invoke.assert_not_called()
        self.assertEqual(self.metadata(output)["stop_reason"], "empty_response")

    def test_does_not_mutate_history_or_accumulate_budget_messages(self):
        model = AgentModelConfig(
            provider="test",
            model="test",
            mode="chat",
            history_prompt_messages=[UserPromptMessage(content="Earlier task")],
        )
        original = deepcopy(model.history_prompt_messages)
        _, prompts, _, _ = self.run_agent(
            [result(calls=[tool_call(1)]), result("Done")],
            params={"model": model},
        )
        self.assertEqual(model.history_prompt_messages, original)
        self.assertNotIn("current_iteration", str(model.history_prompt_messages))
        self.assertEqual(self.budget(prompts[0])["current_iteration"], 1)
        self.assertEqual(self.budget(prompts[1])["current_iteration"], 2)

    def test_parallel_tool_calls_consume_one_round(self):
        output, prompts, _, session = self.run_agent(
            [result(calls=[tool_call(1), tool_call(1, suffix="-b")]), result("Done")],
            params={"maximum_iterations": 2},
        )
        self.assertEqual(len(prompts), 2)
        self.assertEqual(session.tool.invoke.call_count, 2)
        self.assertEqual(self.metadata(output)["iterations_used"], 2)

    def test_invalid_template_is_rejected_before_any_model_call(self):
        session = Mock()
        strategy = FunctionCallingAgentStrategy(runtime=Mock(), session=session)
        with self.assertRaisesRegex(ValueError, "maximum_iterations"):
            list(
                strategy._invoke(
                    {
                        "query": "q",
                        "instruction": "i",
                        "tools": [],
                        "model": AgentModelConfig(provider="test", model="test", mode="chat"),
                        "maximum_iterations": "{{#conversation.limit#}}",
                    }
                )
            )
        session.model.llm.invoke.assert_not_called()

    def test_streaming_budget_and_final_answer(self):
        output, prompts, tools, session = self.run_agent(
            [result(calls=[tool_call(1)]), result("Streamed answer")],
            params={"maximum_iterations": 2},
            streaming=True,
        )
        self.assertIn("Streamed answer", self.text(output))
        self.assertEqual(self.budget(prompts[-1])["current_iteration"], 2)
        self.assertEqual(tools[-1], [])
        self.assertEqual(session.tool.invoke.call_count, 1)


if __name__ == "__main__":
    unittest.main()
