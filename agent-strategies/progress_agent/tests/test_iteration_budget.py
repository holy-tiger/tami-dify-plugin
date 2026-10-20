import unittest

from pydantic import ValidationError

from strategies.iteration_budget import IterationBudget


class TestIterationBudget(unittest.TestCase):
    def test_templates_resolved_to_strings_are_accepted(self):
        budget = IterationBudget(
            maximum_iterations=" 30 ",
            progress_check_start="20",
            progress_check_interval="5",
            final_response_rounds="1",
        )
        self.assertEqual(budget.maximum_iterations, 30)
        self.assertTrue(budget.state(20)["is_checkpoint"])
        self.assertTrue(budget.state(25)["is_checkpoint"])
        self.assertFalse(budget.state(21)["is_checkpoint"])
        self.assertFalse(budget.state(30)["is_checkpoint"])

    def test_exact_remaining_budget(self):
        state = IterationBudget().state(20)
        self.assertEqual(state["current_iteration"], 20)
        self.assertEqual(state["remaining_iterations_after_current"], 10)
        self.assertEqual(state["remaining_tool_rounds_including_current"], 10)
        self.assertFalse(state["is_finalization"])

    def test_last_round_is_reserved(self):
        state = IterationBudget().state(30)
        self.assertTrue(state["is_finalization"])
        self.assertEqual(state["remaining_tool_rounds_including_current"], 0)
        self.assertEqual(state["remaining_iterations_after_current"], 0)

    def test_multiple_reserved_rounds_take_precedence(self):
        budget = IterationBudget(final_response_rounds=6)
        self.assertTrue(budget.state(25)["is_finalization"])
        self.assertFalse(budget.state(25)["is_checkpoint"])

    def test_one_round_budget_is_answer_only(self):
        budget = IterationBudget(maximum_iterations=1)
        self.assertEqual(budget.progress_check_start, 1)
        self.assertTrue(budget.state(1)["is_finalization"])

    def test_small_budget_adjusts_omitted_start(self):
        self.assertEqual(IterationBudget(maximum_iterations=3).progress_check_start, 2)

    def test_explicit_out_of_range_start_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "progress_check_start"):
            IterationBudget(maximum_iterations=3, progress_check_start=20)

    def test_invalid_values_are_rejected_for_every_parameter(self):
        names = (
            "maximum_iterations",
            "progress_check_start",
            "progress_check_interval",
            "final_response_rounds",
        )
        for name in names:
            for value in (
                "",
                "  ",
                "{{#conversation.limit#}}",
                "1.5",
                1.5,
                True,
                None,
                [],
                0,
                -1,
                501,
            ):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValidationError):
                        IterationBudget(**{name: value})

    def test_integer_float_values_are_accepted(self):
        self.assertEqual(IterationBudget(maximum_iterations=30.0).maximum_iterations, 30)

    def test_reserved_rounds_cannot_exceed_total(self):
        with self.assertRaisesRegex(ValidationError, "final_response_rounds"):
            IterationBudget(maximum_iterations=2, final_response_rounds=3)

    def test_invalid_current_round_is_rejected(self):
        for round_number in (0, 31):
            with self.assertRaises(ValueError):
                IterationBudget().state(round_number)


if __name__ == "__main__":
    unittest.main()
