
class BaseTestClass
{
 public:

  //virtual ARCCORE_HOST_DEVICE ~BaseTestClass(){}
  virtual ARCCORE_HOST_DEVICE int apply(int, int) { return 0; }
};
