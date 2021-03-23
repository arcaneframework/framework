// Teste si le compilateur supporte C++14.
// On teste les fonctionnalit�s suivantes:
// - lambda fonctions templates
// - heterogeneous lookup

#include <iostream>
#include <typeinfo>
#include <vector>
#include <string>
#include <set>

void call_test()
{
  auto a = 1.0;
  int z = 5.0;
  std::cout << "TYPE=" << typeid(a).name() << '\n';
  std::cout << "C++_VALUE=" << __cplusplus << '\n';
  auto f = []{ printf("Hello\n"); };
  f();
  auto f2 = [=](int i,int e){ std::cout << "Hello " << i << " e=" << e << " z=" << z << '\n'; };

  z = 3;
  f2(5,12);

  auto f3 = [](auto v){ return v*2; };
  std::cout << f3(1.0) << '\n';
  std::cout << f3(1) << '\n';
}

struct Test {
  Test(std::string name, int id) : m_name(name), m_id(id) {}
  std::string m_name;
  int m_id;
  struct TestCompare {
    using is_transparent = void;
    bool operator()(const Test& lhs, const Test& rhs) const
    {
      return lhs.m_name < rhs.m_name;
    }
    bool operator()(const std::string& name, const Test& ep_id) const
    {
      return name < ep_id.m_name;
    }
    bool operator()(const Test& ep_id, const std::string& name) const
    {
      return name < ep_id.m_name;
    }
  };
};

void call_test2()
{
  std::set<Test, Test::TestCompare> example{{"Toto", 0}, {"Titi", 42}};
  auto search = example.find(42);
  if (search != example.end())
    std::cout  << "Found: " << *search << std::endl;
  else
    std::cout << "Not found..." << std::endl;
};

int
main(int argc,char* argv[])
{
  call_test();
  call_test2();
  return 0;
}
