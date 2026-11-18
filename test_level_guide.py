import dataclasses
from html.parser import HTMLParser
import unittest
from unittest.mock import patch

import app


class LevelGuideParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rules = {}
        self.level = None
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-level-rule" in attrs:
            self.level = int(attrs["data-level-rule"])
            self.rules[self.level] = []
            self.depth = 1
        elif self.level is not None:
            self.depth += 1

    def handle_endtag(self, tag):
        if self.level is not None:
            self.depth -= 1
            if self.depth == 0:
                self.level = None

    def handle_data(self, data):
        if self.level is not None:
            self.rules[self.level].append(data)


class LevelGuideTests(unittest.TestCase):
    def page(self):
        with app.app.test_client() as client:
            response = client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        parser = LevelGuideParser()
        parser.feed(html)
        rules = {level: " ".join(" ".join(text).split()) for level, text in parser.rules.items()}
        return html, rules

    def test_all_five_levels_show_their_configured_brightness(self):
        _, rules = self.page()
        self.assertEqual(set(rules), {0, 1, 2, 3, 4})
        for level in app.AUTO_LEVELS:
            with self.subTest(level=level.level):
                self.assertIn(f"L{level.level} {level.brightness}%", rules[level.level])

    def test_threshold_and_person_boundaries_match_the_existing_rules(self):
        for lux in (None, 0, 269.9, 270):
            for people, expected in ((0, 1), (1, 2), (3, 2), (4, 3), (20, 3)):
                with self.subTest(lux=lux, people=people):
                    self.assertEqual(app.choose_auto_level(people, lux).level, expected)
        for people in (0, 1, 4, 20):
            self.assertEqual(app.choose_auto_level(people, 270.1).level, 0)

    def test_legend_uses_runtime_configuration_instead_of_duplicate_values(self):
        levels = [dataclasses.replace(level, brightness=value)
                  for level, value in zip(app.AUTO_LEVELS, (0, 17, 35, 66, 100))]
        with patch.object(app, "AUTO_LEVELS", levels), \
                patch.object(app, "DAYLIGHT_LUX_THRESHOLD", 412), \
                patch.object(app, "MID_PEOPLE_MAX", 5), \
                patch.object(app, "COUNTDOWN_SECONDS", 23):
            html, rules = self.page()
            self.assertIn("17%", rules[1])
            self.assertIn("35%", rules[2])
            for level in range(4):
                self.assertIn("412 lux", rules[level])
            self.assertIn("1\u20135", rules[2])
            self.assertIn("6 ", rules[3])
            self.assertIn("23", html)
            self.assertEqual(app.choose_auto_level(5, 412).level, 2)
            self.assertEqual(app.choose_auto_level(6, 412).level, 3)

    def test_emergency_rule_shows_the_actual_input_pin_and_active_low_condition(self):
        _, rules = self.page()
        self.assertIn(f"Pin {app.PIN_IN} LOW", rules[4])
        self.assertIn(f"{app.AUTO_LEVELS[4].brightness}%", rules[4])

    def test_guide_is_above_both_application_views(self):
        html, _ = self.page()
        guide_position = html.index('<section class="level-guide"')
        self.assertGreater(guide_position, html.index("</header>"))
        self.assertLess(guide_position, html.index('id="page-home"'))
        self.assertLess(guide_position, html.index('id="page-device"'))


if __name__ == "__main__":
    unittest.main()
