"""The mission library: what the paddock lists as previous missions."""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ergodic_control_mppi.deploy.store import save_spec, stored_specs


class StoreTest(unittest.TestCase):
    def test_saved_specs_list_newest_first_and_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "missions"
            self.assertEqual(stored_specs(directory), [])
            first = save_spec(directory, {"mission_id": "field a"}, datetime(2026, 9, 1, tzinfo=timezone.utc))
            second = save_spec(directory, {"mission_id": "b/../c"}, datetime(2026, 9, 2, tzinfo=timezone.utc))
            self.assertEqual(first, "20260901T000000.000000Z_field_a")
            self.assertNotIn("/", second)
            listed = stored_specs(directory)
            self.assertEqual([name for name, _ in listed], [second, first])
            self.assertEqual(json.loads(listed[1][1]), {"mission_id": "field a"})
            self.assertEqual(len(stored_specs(directory, limit=1)), 1)
            self.assertEqual(sorted(path.name for path in directory.iterdir()), [f"{first}.json", f"{second}.json"])


if __name__ == "__main__":
    unittest.main()
