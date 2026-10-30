"""
HackNEX 2026: On-Device Conversational Stack (HNX26EPS08)
Unit Tests for English Speech Intent & System Actions
Author: Person 2 & Person 1 (Edge Audio & Fast-Path Team)
"""

import unittest
from src.edge.command_analyzer import MultilingualCommandAnalyzer, AnalyzedCommand
from src.edge.router.command_analyzer import analyze_command
from src.edge.router.actions import execute_whitelisted_action


class TestEnglishCommands(unittest.TestCase):
    """Rigorous verification of English intent classification and system action responses."""

    def setUp(self):
        self.analyzer = MultilingualCommandAnalyzer()

    def test_01_battery_intent(self):
        """Battery check commands must parse to BATTERY with real percent output."""
        for phrase in ["check battery", "battery status", "how much battery level"]:
            res = self.analyzer.analyze_and_execute(phrase, execute_actions=False)
            self.assertEqual(res.intent, "BATTERY")
            self.assertIn("Battery is at", res.response_en)

    def test_02_ram_intent(self):
        """RAM check commands must parse to RAM with real memory usage."""
        for phrase in ["check ram", "memory usage", "check system memory"]:
            res = self.analyzer.analyze_and_execute(phrase, execute_actions=False)
            self.assertEqual(res.intent, "RAM")
            self.assertIn("RAM usage is currently at", res.response_en)

    def test_03_notepad_intent(self):
        """Notepad commands must parse to NOTEPAD intent."""
        for phrase in ["open notepad", "launch notepad", "start notepad"]:
            res = self.analyzer.analyze_and_execute(phrase, execute_actions=False)
            self.assertEqual(res.intent, "NOTEPAD")
            self.assertIn("Notepad", res.response_en)

    def test_04_calculator_intent(self):
        """Calculator commands must parse to CALCULATOR intent."""
        for phrase in ["open calculator", "launch calculator", "open calc"]:
            res = self.analyzer.analyze_and_execute(phrase, execute_actions=False)
            self.assertEqual(res.intent, "CALCULATOR")
            self.assertIn("Calculator", res.response_en)

    def test_05_time_intent(self):
        """Time commands must return current formatted clock time."""
        for phrase in ["what time is it", "current time", "tell me the time"]:
            res = self.analyzer.analyze_and_execute(phrase, execute_actions=False)
            self.assertEqual(res.intent, "TIME")
            self.assertIn("The current time is", res.response_en)

    def test_06_date_intent(self):
        """Date commands must return today's calendar date."""
        for phrase in ["what is today's date", "today's date", "what day is it"]:
            res = self.analyzer.analyze_and_execute(phrase, execute_actions=False)
            self.assertEqual(res.intent, "DATE")
            self.assertIn("Today is", res.response_en)

    def test_07_greeting_intent(self):
        """Greetings must return polite voice assistant welcome."""
        for phrase in ["hello", "hi", "good morning", "who are you"]:
            res = self.analyzer.analyze_and_execute(phrase, execute_actions=False)
            self.assertEqual(res.intent, "GREETING")
            self.assertIn("Hello! I am your offline voice assistant", res.response_en)

    def test_08_router_confidence_gate(self):
        """Low confidence utterances (<0.40) must reject action execution."""
        res = analyze_command("open notepad", confidence=0.25, execute_actions=True)
        self.assertFalse(res.action_executed)
        self.assertIn("confidence too low", res.response_en.lower())

    def test_09_unknown_command(self):
        """Unmapped general queries must be marked for LLM forwarding / repetition."""
        res = self.analyzer.analyze_and_execute("what is photosynthesis", execute_actions=False)
        self.assertEqual(res.intent, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
