import unittest
from parliament import Context
from flask import Request
func = __import__("func")

class TestFunc(unittest.TestCase):

  def test_func_empty_request(self):
    resp, code = func.main({})
    self.assertEqual(resp, "{}")
    self.assertEqual(code, 200)

  def test_func_get_request(self):
    req = Request({
      "method": "GET",
    })
    resp, code = func.request_handler(req)
    self.assertIsNotNone(resp)
    self.assertEqual(code, 200)
  
if __name__ == "__main__":
  unittest.main()
  