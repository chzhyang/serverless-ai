import unittest

func = __import__("func")

class TestFunc(unittest.TestCase):

  def test_func_empty_request(self):
    resp, code = func.main({})
    # self.assertEqual(resp, "{}")
    print(resp)
    self.assertEqual(code, 200)
  # def test_func_realdata_request(self):
  #   resp, code = func.main({'input':'data/ILSVRC2012_test_00000002.JPEG'})
  #   # self.assertEqual(resp, "{}")
  #   print(resp)
  #   self.assertEqual(code, 200)

if __name__ == "__main__":
  unittest.main()
  