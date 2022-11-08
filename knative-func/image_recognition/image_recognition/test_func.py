import unittest

func = __import__("func")

class TestFunc(unittest.TestCase):

  def test_func_empty_request(self):
    resp = func.main({})
    # self.assertEqual(resp, "{}")
    # self.assertEqual(code, 400)
    print(resp)

if __name__ == "__main__":
  unittest.main()