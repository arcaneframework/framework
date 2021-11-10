/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <Environment.h>

#include <alien/data/Universal.h>
#include <alien/data/Universe.h>
#include <alien/utils/ObjectWithTrace.h>

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

TEST(TestUniverse, TraceMng)
{
  auto* mng = AlienTest::Environment::traceMng();
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
  auto* mng = AlienTest::Environment::traceMng();
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
  {}
  A(double v)
  : value(v)
  {}
  bool operator==(const A& a) const { return value == a.value; }
  double value;
};

struct B
{
  B()
  : value(0.)
  {}
  B(double v)
  : value(v)
  {}
  bool operator==(const B& b) const { return value == b.value; }
  double value;
};

class C
{
 public:
  C(const A& a ALIEN_UNUSED_PARAM)
  : value(-1.)
  {}
  C(const B& b ALIEN_UNUSED_PARAM)
  : value(-1.)
  {}
  C(const A& a ALIEN_UNUSED_PARAM, const B& b ALIEN_UNUSED_PARAM)
  : value(-1.)
  {}
  C(const B& b ALIEN_UNUSED_PARAM, const A& a ALIEN_UNUSED_PARAM)
  : value(-1.)
  {}
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
  auto* mng = AlienTest::Environment::traceMng();
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
