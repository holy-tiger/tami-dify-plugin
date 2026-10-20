import unittest

from dify_plugin import DifyPluginEnv
from dify_plugin.core.plugin_registration import PluginRegistration

from strategies.function_calling import FunctionCallingAgentStrategy


class TestRegistration(unittest.TestCase):
    def test_sdk_loads_standalone_strategy_and_variable_parameters(self):
        registration = PluginRegistration(DifyPluginEnv())
        self.assertEqual(registration.configuration.author, "tami")
        self.assertEqual(registration.configuration.name, "progress_agent")
        _, strategies = registration.agent_strategies_mapping["progress_agent"]
        configuration, implementation = strategies["progress_function_calling"]
        self.assertEqual(implementation.__name__, FunctionCallingAgentStrategy.__name__)
        parameters = {p.name: p for p in configuration.parameters}
        for name in (
            "maximum_iterations",
            "progress_check_start",
            "progress_check_interval",
            "final_response_rounds",
        ):
            self.assertEqual(parameters[name].type.value, "string")
            self.assertTrue(parameters[name].template.enabled)
        self.assertIn("icon.svg", [asset.filename for asset in registration.files])
