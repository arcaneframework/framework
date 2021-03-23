// Teste si le compilateur supporte C++11.
// On teste les fonctionnalités suivantes:
// - lambda fonctions
// - mot clé 'auto'
// - for range
// - rvalues

#include <iostream>
#include <typeinfo>
#include <vector>

typedef int Integer;
template<typename BodyType> static void
For(Integer i0,Integer size,Integer block_size,const BodyType& bt)
{
  bt(i0,size);
}

// Regarde si les lambda du C++0x sont disponibles
// C'est le cas a partir de icc 11.0 et gcc 4.5
#if defined(__GXX_EXPERIMENTAL_CPP0X__) || defined(__GXX_EXPERIMENTAL_CXX0X__)
// Compilateur intel 11.0 ou +
#if __INTEL_COMPILER && (__INTEL_COMPILER>=1210)
#warning HAS LAMBDA !!
#endif
#endif

// Compilateur GCC 4.7, defini __cplusplus a la bonne valeur (201103)
#if __cplusplus>=201103
#warning GOOD CPLUSPLUS VALUE
#endif

class A
{
 public:
  void f()
  {
    auto f3 = [](int i,int e){ std::cout << "Hello " << i << " e=" << e << '\n'; };
    f3(5,12);
    For(0,100,5,f3);
    For(0,100,5,[](int i,int e){ std::cout << "HelloBoy " << i << " e=" << e << '\n'; } );
  }
  A& operator&(){ std::cout << "Operator&\n"; return *this; }
};

class IFunc
{
 public:
  virtual void apply(int a,int b) =0;
};

template<typename BodyType>
class FuncT
: public IFunc
{
 public:
  FuncT(const BodyType& bd)
  : m_bd(bd){}
  BodyType m_bd;
  virtual void apply(int a,int b)
  {
    std::cout << "** APPLY ME\n";
    m_bd(a,b);
  }
};

template<typename BodyType> FuncT<BodyType>* _createFunc(const BodyType& bd)
{
  return new FuncT<BodyType>(bd);
}

int my_func(IFunc* f)
{
  f->apply(12,32);
  return 0;
}

IFunc* call_test()
{
  auto a = 1.0;
  int z = 5.0;
  std::cout << "TYPE=" << typeid(a).name() << '\n';
  std::cout << "C++_VALUE=" << __cplusplus << '\n';
  auto f = []{ printf("Hello\n"); };
  f();
  auto f2 = [=](int i,int e){ std::cout << "Hello " << i << " e=" << e << " z=" << z << '\n'; };
  IFunc* f4 = _createFunc([&](int i,int e){ std::cout << "Hello " << i << " e=" << e << " z=" << z << '\n'; });
  z = 3;
  f2(5,12);
  f4->apply(9,15);
  std::vector<int> v0;
  int sum = 0;
  for( int x : v0 ){
    sum += x;
  }
  f4->apply(sum,9);
  return f4;
}

// TEST RVALUE
class RVALUE_B
{
 public:
  static RVALUE_B rvalueB() { return RVALUE_B(); }
};

class RVALUE_A
{
 public:
  RVALUE_A(RVALUE_B&& rv){}
};

// ENDTEST
int
main(int argc,char* argv[])
{
  IFunc* ff = call_test();
  my_func(ff);
  A aa;
  aa.f();
  A& aref = &aa;
  return 0;
}
