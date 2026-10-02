"""Truthfulness and lookup tests for the plot capability registry."""

from __future__ import annotations

import unittest

from airwatch.ui.plots.plot_registry import (
    get_signal_view_spec,
    list_signal_view_specs,
)


class PlotRegistryTests(unittest.TestCase):
    def test_registry_keys_are_unique(self) -> None:
        specs = list_signal_view_specs()
        keys = [spec.key for spec in specs]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertGreaterEqual(len(specs), 10)

    def test_workspace_filters_keep_bearing_truthful(self) -> None:
        bearing_specs = list_signal_view_specs(workspace="bearing")
        self.assertIn("waveform", {spec.key for spec in bearing_specs})
        self.assertIn("spectrum", {spec.key for spec in bearing_specs})
        self.assertEqual(
            get_signal_view_spec("bearing_fft").availability,
            "planned",
        )

    def test_constellation_is_migrated_with_explicit_real_compatibility(self) -> None:
        spec = get_signal_view_spec("constellation")
        self.assertEqual(spec.availability, "current")
        self.assertEqual(spec.required_data, "iq_or_explicit_analytic_real")
        self.assertFalse(spec.supports_real)
        self.assertIn("xzt", spec.legacy_location)

    def test_unknown_view_has_actionable_lookup_error(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown signal view"):
            get_signal_view_spec("not-a-real-view")


if __name__ == "__main__":
    unittest.main()
