import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import MagicMock, patch

from paper_roam import PaperRoamError, request_json


class RequestRecoveryTests(unittest.TestCase):
    def test_temporary_network_failure_gets_one_retry(self):
        recovered = MagicMock()
        recovered.__enter__.return_value.read.return_value = b'{"results":[]}'
        with patch("paper_roam.urlopen", side_effect=[URLError("temporary"), recovered]) as call:
            self.assertEqual(request_json("https://api.openalex.org/works", service="OpenAlex"), {"results": []})
        self.assertEqual(call.call_count, 2)

    def test_persistent_failure_stops_after_the_second_attempt(self):
        with patch("paper_roam.urlopen", side_effect=URLError("private detail")) as call:
            with self.assertRaises(PaperRoamError) as failure:
                request_json("https://api.openalex.org/works", service="OpenAlex")
        self.assertEqual(call.call_count, 2)
        self.assertNotIn("private detail", str(failure.exception))

    def test_unauthorized_stops_without_retry(self):
        refused = HTTPError("https://api.openalex.org/works", 401, "Unauthorized", {}, None)
        with patch("paper_roam.urlopen", side_effect=refused) as call:
            with self.assertRaisesRegex(PaperRoamError, "HTTP 401"):
                request_json("https://api.openalex.org/works", service="OpenAlex")
        self.assertEqual(call.call_count, 1)
