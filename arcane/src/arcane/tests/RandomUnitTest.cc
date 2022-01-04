// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RandomUnitTest.cc                                           (C) 2000-2014 */
/*                                                                           */
/* Test des générateurs aléatoires.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/random/Uniform01.h"
#include "arcane/random/UniformOnSphere.h"
#include "arcane/random/LinearCongruential.h"
#include "arcane/random/InversiveCongruential.h"
#include "arcane/random/TMrg32k3a.h"
#include "arcane/random/TKiss.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/FactoryService.h"

#include "arcane/tests/ArcaneTestGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test des ItemVector
 */
class RandomUnitTest
: public BasicUnitTest
{
 public:

  RandomUnitTest(const ServiceBuildInfo& cb);
  ~RandomUnitTest();

 public:

  virtual void initializeTest() {}
  virtual void executeTest();

 private:

  template<typename RngType> void
  _checkGenerator(RngType& rng,const String& str);
  template<typename RngType> void
  _checkGeneratorUniform(RngType& rng,const String& str);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(RandomUnitTest,IUnitTest,RandomUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RandomUnitTest::
RandomUnitTest(const ServiceBuildInfo& mb)
: BasicUnitTest(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RandomUnitTest::
~RandomUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename RngType>
void RandomUnitTest::
_checkGeneratorUniform(RngType& rng,const String& name)
{
  random::Uniform01<RngType> uniform_rng(rng);
  info() << "Checking uniform generation for generator class=" << name;
  for( Integer i=0; i<1000000; ++i ){
    Real z = uniform_rng();
    if (z>=1.0 || z<0.0)
      fatal() << "Invalid generated value (1)" << z;

    Real z2 = random::Uniform01<RngType>::apply(rng,rng());
    if (z2>=1.0 || z2<0.0)
      fatal() << "Invalid generated value (2)" << z2;
  }
  Real3 r = random::UniformOnSphere<random::Uniform01<RngType> >(uniform_rng).applyDim3();
  info() << "R=" << r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename RngType>
void RandomUnitTest::
_checkGenerator(RngType& rng,const String& name)
{
  info() << "Checking generator class=" << name;
  for( Integer i=0; i<50; ++i ){
    typename RngType::result_type state = rng();
    info() << " STATE=" << state;
  }

  _checkGeneratorUniform(rng,name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RandomUnitTest::
executeTest()
{
  {
    // LinearCongruential
    random::MinstdRand mr(53);
    _checkGenerator(mr,"MinstdRand");
  }
  // TODO pouvoir activer le test et trapper l'exception
  // Faire de meme avec la valeur 0.
#if 0
  {
    random::MinstdRand mr(INT_MAX);
    _checkGenerator(mr,"MinstdRand(INT_MAX)");
  }
#endif

  {
    // LinearCongruential
    random::MinstdRand0 mr0(72);
    _checkGenerator(mr0,"MinstdRand0");
  }

  // TODO pouvoir activer le test et trapper l'exception
#if 0
  {
    // LinearCongruential
    random::MinstdRand0 mr0(INT_MAX);
    _checkGenerator(mr0,"MinstdRand0(INT_MAX)");
  }
#endif

  {
    // InversiveCongruential
    random::Hellekalek1995 hk(45214);
    _checkGenerator(hk,"Hellekalek1995");
  }
  {
    // TMrg32k3a
    Real state[] = { 4512,412,2131,145,1234,63463 };
    random::Mrg32k3a mrg(state);
    _checkGenerator(mrg,"Mrg32k3a");
  }
  {
    // TKiss
    random::Kiss kiss(4512,932,31532915,1234,63463);
    _checkGenerator(kiss,"Kiss");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
