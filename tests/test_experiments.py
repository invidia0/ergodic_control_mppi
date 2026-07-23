import csv
import tempfile
import unittest
from pathlib import Path

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.experiments.bo import BO_CSV_FIELDS
from ergodic_control_mppi.experiments.common import (
    append_csv,
    prepare_outputs,
    variant_from_mapping,
)
from ergodic_control_mppi.parameters import apply_variant


class ExperimentsTest(unittest.TestCase):
    def test_typed_variant_uses_nested_replacement(self):
        params = load_config("configs/mppi_params.yaml").controller
        variant = variant_from_mapping({"weight_stein": 12.0, "horizon": 7})
        changed = apply_variant(params, variant)
        self.assertEqual(changed.stein.flow_weight, 12.0)
        self.assertEqual(changed.mppi.horizon, 7)
        self.assertIs(changed.workspace, params.workspace)

    def test_unknown_variant_field_is_rejected(self):
        with self.assertRaises(ValueError):
            variant_from_mapping({"unused": 1})

    def test_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.csv"
            output.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_outputs([output], overwrite=False)
            prepare_outputs([output], overwrite=True)
            self.assertFalse(output.exists())

    def test_csv_header_and_bo_error_field_contract(self):
        self.assertIn("error", BO_CSV_FIELDS)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "row.csv"
            append_csv(output, {"a": 1, "b": 2}, ["a", "b"])
            with output.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(next(csv.reader(stream)), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
