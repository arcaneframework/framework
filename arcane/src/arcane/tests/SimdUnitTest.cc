// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimdUnitTest.cc                                             (C) 2000-2016 */
/*                                                                           */
/* Service de test des classes gérant la vectorisation.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/ValueChecker.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/FactoryService.h"

#include "arcane/utils/Simd.h"
#include "arcane/utils/SimdOperation.h"

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
 * \brief Service de test des classes de vectorisation.
 *
 * \todo tester tous les mécanismes de vectorisation accessible sur la
 * plateforme et pas seulement celui par défaut (par exemple sur KNL, tester
 * SSE, AVX et AVX512)
 */
class SimdUnitTest
: public BasicUnitTest
{
 public:

  SimdUnitTest(const ServiceBuildInfo& cb);
  ~SimdUnitTest();

 public:

  void initializeTest() override {}
  void executeTest() override;

 private:

  void _testSimdArray();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ensemble de valeurs pour les tests SIMD.
 *
 * \note Pour garantir le respect de l'alignement mémoire des classes Simd,
 * les instances de cette classe ne doivent être allouées que sur la pile.
 */
template<typename SimdRealType>
class SimdTestValue
{
 public:

  void initialize(ITraceMng* tm);

 public:

  const SimdRealType& simdA() const { return m_simd_a; }
  const SimdRealType& simdB() const { return m_simd_b; }

  ConstArrayView<Real> scalarA() const { return m_scalar_a; }
  ConstArrayView<Real> scalarB() const { return m_scalar_b; }

  ConstArrayView<Real> simdAsArrayA() const { return m_simd_as_array_a; }
  ConstArrayView<Real> simdAsArrayB() const { return m_simd_as_array_b; }

 public:

  SimdRealType m_simd_a;
  SimdRealType m_simd_b;

  ArrayView<Real> m_scalar_a;
  ArrayView<Real> m_scalar_b;
  UniqueArray<Real> m_scalar_array;

  ArrayView<Real> m_simd_as_array_a;
  ArrayView<Real> m_simd_as_array_b;
  UniqueArray<Real> m_simd_as_array_array;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimdRealType>
void SimdTestValue<SimdRealType>::
initialize(ITraceMng* tm)
{
  // m_scalar_a et m_scalar_b contiennent les valeurs de référence sous
  // forme de tableau de scalaires.
  Integer vec_length = SimdRealType::Length;
  m_scalar_array.resize(vec_length*2);
  m_scalar_a = m_scalar_array.subView(0,vec_length);
  m_scalar_b = m_scalar_array.subView(vec_length,vec_length);

  m_simd_as_array_array.resize(vec_length*2);
  m_simd_as_array_a = m_simd_as_array_array.subView(0,vec_length);
  m_simd_as_array_b = m_simd_as_array_array.subView(vec_length,vec_length);

  for( Integer i=0; i<vec_length; ++i ){
    m_scalar_a[i] = 1.0;
    m_scalar_b[i] = 2.0;

    m_simd_as_array_a[i] = (Real)(i+1);
    m_simd_as_array_b[i] = (Real)(i+15);

    m_simd_a[i] = m_simd_as_array_a[i];
    m_simd_b[i] = m_simd_as_array_b[i];

    tm->info() << "SimdA i=" << i << " = " << m_simd_a[i];
    tm->info() << "SimdB i=" << i << " = " << m_simd_b[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(SimdUnitTest,IUnitTest,SimdUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class BinaryOperator,typename SimdRealType>
class SimdBinaryOperatorTester
{
  static void _checkDirect(const SimdRealType& simd_result,ConstArrayView<Real> ref_a,
                           ConstArrayView<Real> ref_b,const String& msg)
  {
    for( Integer i=0, n=SimdRealType::Length; i<n; ++i ){
      Real v = BinaryOperator::apply(ref_a[i],ref_b[i]);
      Real r = simd_result[i];
      if (v!=r)
        ARCANE_FATAL("Bad Value value='{0}' expected='{1}' test={2} index={3} ref_a={4} ref_b={5}",
                     r,v,msg,i,ref_a[i],ref_b[i]);
    }
  }

 public:

  static void doTest(const SimdUnitTest& st,const SimdTestValue<SimdRealType>& sv,const String& op_name)
  {
    SimdRealType ra = sv.simdA();
    SimdRealType rb = sv.simdB();
    ConstArrayView<Real> ref_a = sv.simdAsArrayA();
    ConstArrayView<Real> ref_b = sv.simdAsArrayB();
    Real a = sv.scalarA()[0];
    Real b = sv.scalarB()[0];

    SimdRealType result0 = BinaryOperator::apply(a,rb);
    _checkDirect(result0,sv.scalarA(),ref_b,"1");

    SimdRealType result1 = BinaryOperator::apply(ra,b);
    _checkDirect(result1,ref_a,sv.scalarB(),"2");

    SimdRealType result2 = BinaryOperator::apply(ra,rb);
    _checkDirect(result2,ref_a,ref_b,"3");

    st.info() << "Check OK for " << op_name;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class UnaryOperator,typename SimdRealType>
class SimdUnaryOperatorTester
{
  static void _checkDirect(const SimdRealType& simd_result,
                           ConstArrayView<Real> ref_a,const String& msg)
  {
    for( Integer i=0, n=SimdRealType::Length; i<n; ++i ){
      Real v = UnaryOperator::apply(ref_a[i]);
      Real r = simd_result[i];
      if (v!=r)
        ARCANE_FATAL("Bad Value value='{0}' expected='{1}' test={2} index={3} ref_a={4}",
                     r,v,msg,i,ref_a[i]);
    }
  }

 public:

  static void doTest(const SimdUnitTest& st,const SimdTestValue<SimdRealType>& sv,
                     const String& op_name)
  {
    const SimdRealType& ra = sv.simdA();
    ConstArrayView<Real> ref_a = sv.simdAsArrayA();

    SimdRealType result = UnaryOperator::apply(ra);
    _checkDirect(result,ref_a,"2");

    st.info() << "Check OK for " << op_name;
  }
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/SimdGeneratedUnitTest.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimdUnitTest::
SimdUnitTest(const ServiceBuildInfo& mb)
: BasicUnitTest(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimdUnitTest::
~SimdUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimdInfoType>
class SimdTester
{
 public:

  typedef typename SimdInfoType::SimdReal SimdRealType;

 public:

  static void test(const SimdUnitTest& st)
  {
    st.info() << "Testing SimdTest type=" << SimdInfoType::name();
    _doCopyTest(st);
    _doOperatorTest(st);
    _doFMATest(st);
  }

 private:

  static void _doOperatorTest(const SimdUnitTest& st)
  {
    SimdTestValue<SimdRealType> simd_value;
    simd_value.initialize(st.traceMng());
    _doAllBinary(st,simd_value);
    _doAllUnary(st,simd_value);
  }

  static void _doCopyTest(const SimdUnitTest& st)
  {
    ValueChecker vc(A_FUNCINFO);
    vc.setThrowOnError(false);

    {
      SimdRealType simd_real_array[6];
      for( Integer i=0; i<6; ++i ){
        Real v = 3.0 * i;
        SimdRealType r(v);
        for( Integer z=0; z<SimdRealType::BLOCK_SIZE; ++z )
          vc.areEqual(r[z],v,"SimdReal constructor");
        simd_real_array[i] = r;
        st.info() << "V1 i=" << i << " v=" << simd_real_array[i] << " r=" << r;
        st.info() << " r=" << r;
        for( Integer z=0; z<SimdRealType::BLOCK_SIZE; ++z )
          vc.areEqual(simd_real_array[i][z],v,"fixed SimdReal[]");
      }
    }

    {
      // Les tableaux de type 'SimdReal' doivent être alignés
      // sur un multiple de 64 avec l'AVX512 (qui est le plus restrictif).
      // TODO: gérer cela dans la classe Array.
      int n = 17;
      char buf[(17 * sizeof(SimdRealType)) + 64];
      void* aligned_buf = nullptr;
      // Utilise 'buf' comme tableau mémoire temporaire
      // et récupère depuis buf un bloc aligné sur 32 octets
      // qu'on stocke dans 'aligned_buf'
      for( Integer i=0; i<64; i+=8 ){
        void* xbuf = buf + i;
        Int64 p = (Int64)(xbuf);
        if ((p % 32)==0){
          aligned_buf = xbuf;
          break;
        }
      }

      if (aligned_buf){
        SimdRealType* asimd_real_array = (SimdRealType*)aligned_buf;

        for( Integer i=0; i<n; ++i ){
          Real v = 3.0 * i;
          SimdRealType r(v);
          asimd_real_array[i] = r;
          st.info() << "V2 i=" << i << " v=" << asimd_real_array[i] << " r=" << r;
          st.info() << " r=" << r;
          for( Integer z=0; z<SimdRealType::BLOCK_SIZE; ++z )
            vc.areEqual(asimd_real_array[i][z],v,"SimdReal*");
        }
      }
    }

    vc.throwIfError();
  }
  static void _doFMATest(const SimdUnitTest& st);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimdInfoType>
void SimdTester<SimdInfoType>::
_doFMATest(const SimdUnitTest& st)
{
  SimdRealType a, b, c, d;
  for( Integer z=0, n=SimdRealType::BLOCK_SIZE; z<n; ++z ){
    a[z] = 1.0 * z;
    b[z] = 3.0 * z;
    c[z] = 6.0 * z;
    d[z] = 9.0 * z;
  }

  // Si le compilateur est performant, cette instruction doit pouvoir
  // être remplacée par un FMA. C'est le cas avec gcc à partir de la
  // version 5.
  d = (a * b) + c;

  st.info() << "PrintD=" << d;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimdUnitTest::
_testSimdArray()
{
  info() << "TEST SIMD ARRAY";
  Integer wanted_size = 19;
  UniqueArray<Real> a(AlignedMemoryAllocator::Simd(),wanted_size);
  UniqueArray<Real> b(AlignedMemoryAllocator::Simd(),wanted_size);
  UniqueArray<Real> c(AlignedMemoryAllocator::Simd(),wanted_size);
  UniqueArray<Real> d(AlignedMemoryAllocator::Simd(),wanted_size);
  UniqueArray<Real> expected_result(wanted_size);

  for( Integer i=0; i<wanted_size; ++i ){
    a[i] = 1.0;
    b[i] = 3.0 + (Real)i;
    c[i] = 7.0 + (Real)(i*2);
    // L'expression suivante doit être la même que pour l'itération Simd
    // suivante.
    expected_result[i] = b[i] + c[i] * a[i];
  }
  applySimdPadding(a);
  applySimdPadding(b);
  applySimdPadding(c);
  Integer simd_size = SimdReal::BLOCK_SIZE;
  Integer padding_size = arcaneSizeWithPadding(wanted_size);
  Integer nb_iteration = padding_size / simd_size;
  info() << "size=" << wanted_size << " padding_size=" << padding_size
         << " simd_size=" << simd_size
         << " nb_iteration=" << nb_iteration;
  SimdReal* xa = (SimdReal*)a.unguardedBasePointer();
  SimdReal* xb = (SimdReal*)b.unguardedBasePointer();
  SimdReal* xc = (SimdReal*)c.unguardedBasePointer();
  SimdReal* xd = (SimdReal*)d.unguardedBasePointer();
  for( Integer i=0; i<nb_iteration; ++i ){
    xd[i] = xb[i] + xc[i] * xa[i];
  }
  ValueChecker vc(A_FUNCINFO);
  vc.areEqualArray(d,expected_result,"SimdLoop");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimdUnitTest::
executeTest()
{
  SimdTester<SimdInfo>::test(*this);
  SimdTester<EMULSimdInfo>::test(*this);
#ifdef ARCANE_HAS_AVX512
  SimdTester<AVX512SimdInfo>::test(*this);
#endif
#ifdef ARCANE_HAS_AVX
  SimdTester<AVXSimdInfo>::test(*this);
#endif
#ifdef ARCANE_HAS_SSE
  SimdTester<SSESimdInfo>::test(*this);
#endif
  _testSimdArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
