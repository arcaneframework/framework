#include "gtest/gtest.h"

#include <alien/Alien.h>

namespace Environment {
extern Arccore::ITraceMng* traceMng();
}

struct Tracer : public Alien::ObjectWithTrace
{
  Tracer() {}

  void printInfo()
  {
    alien_info([&] { cout() << "Info"; });
  }

  void printDebug()
  {
    alien_debug([&] { cout() << "Debug"; });
  }

  void printWarning()
  {
    alien_warning([&] { cout() << "Warning"; });
  }
};

TEST(TestUniverse, Reset)
{
  Alien::Universe u;
  u.reset();
  ASSERT_TRUE(u.traceMng() == nullptr);
}

TEST(TestUniverse, NoTraceMng)
{
  Alien::Universe u;
  ASSERT_TRUE(u.traceMng() == nullptr);
  u.reset();
}

TEST(TestUniverse, TraceMng)
{
  auto* mng = Environment::traceMng();
  Alien::Universe u;
  u.setTraceMng(mng);
  auto* trace = u.traceMng();
  ASSERT_TRUE(trace != nullptr);
  Alien::Universe u2;
  auto* trace2 = u2.traceMng();
  ASSERT_TRUE(trace2 != nullptr);
  ASSERT_TRUE(trace == trace2);
  u.reset();
  ASSERT_TRUE(u.traceMng() == nullptr);
  u.reset();
}

TEST(TestUniverse, ObjectWithTrace)
{
  auto* mng = Environment::traceMng();
  Alien::setTraceMng(mng);
  Alien::setVerbosityLevel(Alien::Verbosity::Debug);
  Tracer tracer;
  std::cout << "Verbosity Debug: \n";
  tracer.printInfo();
  tracer.printDebug();
  tracer.printWarning();
  Alien::setVerbosityLevel(Alien::Verbosity::Info);
  std::cout << "Verbosity Info: \n";
  tracer.printInfo();
  tracer.printDebug();
  tracer.printWarning();
  std::cout << "Verbosity Warning: \n";
  Alien::setVerbosityLevel(Alien::Verbosity::Warning);
  tracer.printInfo();
  tracer.printDebug();
  tracer.printWarning();
  Alien::Universe().reset();
}

struct A
{
  A()
  : value(0.)
  {
  }
  A(double v)
  : value(v)
  {
  }
  bool operator==(const A& a) const { return value == a.value; }
  double value;
};

struct B
{
  B()
  : value(0.)
  {
  }
  B(double v)
  : value(v)
  {
  }
  bool operator==(const B& b) const { return value == b.value; }
  double value;
};

class C
{
 public:
  C(const A& a ALIEN_UNUSED_PARAM)
  : value(-1.)
  {
  }
  C(const B& b ALIEN_UNUSED_PARAM)
  : value(-1.)
  {
  }
  C(const A& a ALIEN_UNUSED_PARAM, const B& b ALIEN_UNUSED_PARAM)
  : value(-1.)
  {
  }
  C(const B& b ALIEN_UNUSED_PARAM, const A& a ALIEN_UNUSED_PARAM)
  : value(-1.)
  {
  }
  ~C() {}
  double value;
};

TEST(TestUniverse, DataBase)
{
  Alien::Universe u;
  auto& db = u.dataBase();
  A a;
  B b;
  auto la = db.findOrCreate<C>(a);
  ASSERT_TRUE(la.second);
  auto lb = db.findOrCreate<C>(b);
  ASSERT_TRUE(lb.second);
  auto lab = db.findOrCreate<C>(a, b);
  ASSERT_TRUE(lab.second);
  auto lba = db.findOrCreate<C>(b, a);
  ASSERT_TRUE(lba.second);
  auto la2 = db.findOrCreate<C>(a);
  ASSERT_FALSE(la2.second);
  ASSERT_EQ(la.first, la2.first);
  u.reset();
}

TEST(TestUniverse, Universal)
{
  auto* mng = Environment::traceMng();
  Alien::setTraceMng(mng);
  Alien::setVerbosityLevel(Alien::Verbosity::Debug);
  {
    A a(1.);
    B b(2.);
    Alien::Universal<C> c(a, b);
    c->value = 3.14;
  }
  {
    A a(2.);
    B b(2.);
    Alien::Universal<C> c(a, b);
    ASSERT_EQ(-1., c->value);
  }
  {
    A a(1.);
    B b(2.);
    Alien::Universal<C> c(a, b);
    ASSERT_EQ(3.14, c->value);
  }
  {
    A a(0.);
    B b(2.);
    Alien::Universal<C> c(a, b);
    c.first_time([](C& c) { c.value = 5.01; });
    ASSERT_EQ(5.01, c->value);
  }
  {
    A a(0.);
    B b(2.);
    Alien::Universal<C> c(a, b);
    c.first_time([](C& c) { c.value = 5.01; });
    ASSERT_EQ(5.01, c->value);
  }
}
