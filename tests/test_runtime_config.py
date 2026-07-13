import unittest

from runtime_config import LaunchConfig, load_launch_config


class RuntimeConfigTests(unittest.TestCase):
    def test_defaults_to_localhost_without_a_public_share_link(self):
        self.assertEqual(
            load_launch_config({}),
            LaunchConfig(server_name="127.0.0.1", server_port=7860, share=False),
        )

    def test_accepts_explicit_network_configuration(self):
        config = load_launch_config(
            {
                "DARIJA_DEMO_HOST": " 0.0.0.0 ",
                "DARIJA_DEMO_PORT": " 8080 ",
                "DARIJA_DEMO_SHARE": " TRUE ",
            }
        )
        self.assertEqual(
            config,
            LaunchConfig(server_name="0.0.0.0", server_port=8080, share=True),
        )

    def test_accepts_documented_boolean_values(self):
        for raw_value in ("1", "true", "TRUE", "yes", "on"):
            with self.subTest(raw_value=raw_value):
                config = load_launch_config({"DARIJA_DEMO_SHARE": raw_value})
                self.assertTrue(config.share)
        for raw_value in ("0", "false", "FALSE", "no", "off"):
            with self.subTest(raw_value=raw_value):
                config = load_launch_config({"DARIJA_DEMO_SHARE": raw_value})
                self.assertFalse(config.share)

    def test_rejects_ambiguous_share_value(self):
        with self.assertRaisesRegex(ValueError, "DARIJA_DEMO_SHARE must be one of"):
            load_launch_config({"DARIJA_DEMO_SHARE": "sometimes"})

    def test_rejects_invalid_ports(self):
        invalid_ports = {
            "not-a-number": "must be an integer",
            "": "must be an integer",
            "0": "must be between 1 and 65535",
            "65536": "must be between 1 and 65535",
        }
        for raw_value, message in invalid_ports.items():
            with self.subTest(raw_value=raw_value):
                with self.assertRaisesRegex(ValueError, message):
                    load_launch_config({"DARIJA_DEMO_PORT": raw_value})

    def test_rejects_blank_host(self):
        with self.assertRaisesRegex(ValueError, "DARIJA_DEMO_HOST must not be blank"):
            load_launch_config({"DARIJA_DEMO_HOST": "  "})


if __name__ == "__main__":
    unittest.main()
