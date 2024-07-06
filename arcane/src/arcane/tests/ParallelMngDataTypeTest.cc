// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelMngDataTypeTest.cc                                  (C) 2000-2023 */
/*                                                                           */
/* Test des opérations de base du parallèlisme.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/HPReal.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Array2.h"
#include "arcane/utils/Array3View.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/ValueChecker.h"

#include "arcane/IParallelMng.h"
#include "arcane/IParallelNonBlockingCollective.h"

#include "arcane/tests/ArcaneTestGlobal.h"

#include "arcane/parallel/IRequestList.h"

#include "arccore/message_passing/Messages.h"
#include "arccore/message_passing/GatherMessageInfo.h"

#include "arcane/datatype/DataTypeTraits.h"

#include <cstdint>
#include <thread>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace {

// Classe template permettant d'indique si 'DataType' a une implémentation dans
// 'Arccore::MessagePassing::IMessagePassingMng'. Pour l'instant c'est le cas
// uniquement pour les types de base.
template<typename DataType>
class HasMessagePassingMngImplementation
{
 public:
  static constexpr bool hasImpl() { return true; }
};

template<>
class HasMessagePassingMngImplementation<Arcane::Real2>
{
 public:
  static constexpr bool hasImpl() { return false; }
};
template<>
class HasMessagePassingMngImplementation<Arcane::Real3>
{
 public:
  static constexpr bool hasImpl() { return false; }
};
template<>
class HasMessagePassingMngImplementation<Arcane::Real2x2>
{
 public:
  static constexpr bool hasImpl() { return false; }
};
template<>
class HasMessagePassingMngImplementation<Arcane::Real3x3>
{
 public:
  static constexpr bool hasImpl() { return false; }
};
template<>
class HasMessagePassingMngImplementation<Arcane::HPReal>
{
 public:
  static constexpr bool hasImpl() { return false; }
};

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Parallel;
namespace MP = Arccore::MessagePassing;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>
class ParallelMngDataTypeValueGenerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Pour les Int16, les cas tests utilisent au plus 12 coeurs.
// Pour les reductions de type 'somme', on fait en sorte que chaque valeur
// ne dépasse pas 32768 / 12 pour éviter de faire des débordements de valeur
// dont le comportement n'est pas défini.
template<>
class ParallelMngDataTypeValueGenerator<Int16>
{
 public:
  static Int16 zero() { return 0; }
  static Int16 generateTriValue(Int32 v1,Int32 v2,Int32 v3)
  {
    return static_cast<Int16>(v1 + v2 + v3 + 1) % 2563;
  }
  static Int16 generateBiValue(Int32 v1,Int32 v2)
  {
    return static_cast<Int16>(v1 + v2 + 3) % 2563;
  }
  static Int16 generateMonoValue(Int32 v)
  {
    return static_cast<Int16>(4 + 2 * ((v-5)*(v-5))) % 2563;
  }
  static bool isMultiReal() { return false; }
};

template<>
class ParallelMngDataTypeValueGenerator<Int32>
{
 public:
  static Int32 zero() { return 0; }
  static Int32 generateTriValue(Int32 v1,Int32 v2,Int32 v3)
  {
    return v1 + v2 + v3 + 1;
  }
  static Int32 generateBiValue(Int32 v1,Int32 v2)
  {
    return v1 + v2 + 3;
  }
  static Int32 generateMonoValue(Int32 v)
  {
    return 4 + 2 * ((v-5)*(v-5));
  }
  static bool isMultiReal() { return false; }
};

template<>
class ParallelMngDataTypeValueGenerator<Int64>
{
 public:
  static Int64 zero() { return 0; }
  static Int64 generateTriValue(Int32 v1,Int32 v2,Int32 v3)
  {
    return v1 + v2 + v3 + 1;
  }
  static Int64 generateBiValue(Int32 v1,Int32 v2)
  {
    return v1 + v2 + 3;
  }
  static Int64 generateMonoValue(Int32 v)
  {
    Int64 xv = (v-5);
    return 4 + 2 * (xv*xv);
  }
  static bool isMultiReal() { return false; }
};

template<>
class ParallelMngDataTypeValueGenerator<Real>
{
 public:
  static Real zero() { return 0.0; }
  static Real generateTriValue(Int32 v1,Int32 v2,Int32 v3)
  {
    return (double)v1 + (double)v2 + (double)v3 + 1.0;
  }
  static Real generateBiValue(Int32 v1,Int32 v2)
  {
    return (double)v1 + (double)v2 + 3.0;
  }
  static Real generateMonoValue(Int32 v)
  {
    return 2.0 + 0.5 * (double)((v-5)*(v-5));
  }
  static bool isMultiReal() { return false; }
};

template<>
class ParallelMngDataTypeValueGenerator<Real2>
{
 public:
  static Real2 zero() { return Real2::null(); }
  static Real2 generateTriValue(Int32 v1,Int32 v2,Int32 v3)
  {
    Real x = (double)v1 + (double)v2 + (double)v3 + 1.0;
    return Real2(x,x+1.0);
  }
  static Real2 generateBiValue(Int32 v1,Int32 v2)
  {
    Real x = (double)v1 + (double)v2 + 3;
    return Real2(x,x+1.0);
  }
  static Real2 generateMonoValue(Int32 v)
  {
    Real x = 2.0 + 0.5 * (double)((v-5)*(v-5));
    return Real2(x,x+1.0);
  }
  static bool isMultiReal() { return true; }
};

template<>
class ParallelMngDataTypeValueGenerator<Real3>
{
 public:
  static Real3 zero() { return Real3::null(); }
  static Real3 generateTriValue(Int32 v1,Int32 v2,Int32 v3)
  {
    Real x = (double)v1 + (double)v2 + (double)v3 + 1.0;
    return Real3(x,x+1.0,x+2.0);
  }
  static Real3 generateBiValue(Int32 v1,Int32 v2)
  {
    Real x = (double)v1 + (double)v2 + 3;
    return Real3(x,x+1.0,x+2.0);
  }
  static Real3 generateMonoValue(Int32 v)
  {
    Real x = 2.0 + 0.5 * (double)((v-5)*(v-5));
    return Real3(x,x+1.0,x+2.0);
  }
  static bool isMultiReal() { return true; }
};

template<>
class ParallelMngDataTypeValueGenerator<Real2x2>
{
 public:
  static Real2x2 zero() { return Real2x2::null(); }
  static Real2x2 generateTriValue(Int32 v1,Int32 v2,Int32 v3)
  {
    Real x = (double)v1 + (double)v2 + (double)v3 + 1.0;
    return Real2x2::fromLines(x,x+1.0,x+2.0,x+3.0);
  }
  static Real2x2 generateBiValue(Int32 v1,Int32 v2)
  {
    Real x = (double)v1 + (double)v2 + 3;
    return Real2x2::fromLines(x,x+1.0,x+2.0,x+3.0);
  }
  static Real2x2 generateMonoValue(Int32 v)
  {
    Real x = 2.0 + 0.5 * (double)((v-5)*(v-5));
    return Real2x2::fromLines(x,x+1.0,x+2.0,x+3.0);
  }
  static bool isMultiReal() { return true; }
};

template<>
class ParallelMngDataTypeValueGenerator<Real3x3>
{
 public:
  static Real3x3 zero() { return Real3x3::null(); }
  static Real3x3 generateTriValue(Int32 v1,Int32 v2,Int32 v3)
  {
    Real x = (double)v1 + (double)v2 + (double)v3 + 1.0;
    return Real3x3::fromLines(x,x+1.0,x+2.0,x+3.0,
                              x+4.0,x+5.0,x+6.0,x+7.0,x+8.0);
  }
  static Real3x3 generateBiValue(Int32 v1,Int32 v2)
  {
    Real x = (double)v1 + (double)v2 + 3;
    return Real3x3::fromLines(x,x+1.0,x+2.0,x+3.0,
                              x+4.0,x+5.0,x+6.0,x+7.0,x+8.0);
  }
  static Real3x3 generateMonoValue(Int32 v)
  {
    Real x = 2.0 + 0.5 * (double)((v-5)*(v-5));
    return Real3x3::fromLines(x,x+1.0,x+2.0,x+3.0,
                              x+4.0,x+5.0,x+6.0,x+7.0,x+8.0);
  }
  static bool isMultiReal() { return true; }
};

template<>
class ParallelMngDataTypeValueGenerator<HPReal>
{
  // On utilise un log pour avoir des valeurs avec
  // une précision suffisante pour générer des
  // erreur d'arrondi lors de la somme.
 public:
  static HPReal zero() { return HPReal::zero(); }
  static HPReal generateTriValue(Int32 v1,Int32 v2,Int32 v3)
  {
    return HPReal(math::log((double)(v1 + v2 + v3 + 1)));
  }
  static HPReal generateBiValue(Int32 v1,Int32 v2)
  {
    return HPReal(math::log((double)(v1 + v2 + 3)));
  }
  static HPReal generateMonoValue(Int32 v)
  {
    return HPReal(math::log(2.0 + 0.5 * (double)((v+5)*(v+5))));
  }
  static bool isMultiReal() { return false; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour tester les appels IParallelMng pour le type \a DataType.
 */
template<typename DataType>
class ParallelMngDataTypeTest
: public TraceAccessor
{
 public:

  struct CommInfo
  {
   public:
    UniqueArray<DataType> ref_values; //!< Valeurs de référence
    UniqueArray<DataType> send_values; //!< Valeurs à envoyer.
    UniqueArray<DataType> recv_values; //!< Valeurs à recevoir.
  };

  struct AllToAllVariableCommInfo
  {
   public:
    UniqueArray<DataType> ref_values; //!< Valeurs de référence
    UniqueArray<DataType> send_values; //!< Valeurs à envoyer.
    UniqueArray<DataType> recv_values; //!< Valeurs à recevoir.

    UniqueArray<Int32> send_count;
    UniqueArray<Int32> send_index;
    UniqueArray<Int32> recv_count;
    UniqueArray<Int32> recv_index;
  };

 public:

  struct MinMaxSumInfo
  {
    MinMaxSumInfo(const DataType& _min_val,const DataType& _max_val,
                  const DataType& _sum_val,
                  Int32 _min_rank,Int32 _max_rank)
    : min_val(_min_val), max_val(_max_val), sum_val(_sum_val),
      min_rank(_min_rank), max_rank(_max_rank) { }
   public:
    DataType min_val;
    DataType max_val;
    DataType sum_val;
    Int32 min_rank;
    Int32 max_rank;
  };
 public:
  typedef ParallelMngDataTypeTest ThatClass;
  typedef ParallelMngDataTypeValueGenerator<DataType> Generator;
 public:
  ParallelMngDataTypeTest(IParallelMng* pm,const String& datatype_name)
  : TraceAccessor(pm->traceMng()), m_parallel_mng(pm),
    m_datatype_name(datatype_name), m_verbose(false)
  {
  }
 public:
  void doTests();
 private:
  void _launchTest(const String& name,void (ParallelMngDataTypeTest::*func)())
  {
    ITraceMng* tm = m_parallel_mng->traceMng();
    tm->info() << "Test " << name;
    (this->*func)();
    tm->info() << "Test " << name << " finished";
    tm->flush();
  }
  void _testComputeMinMaxSum();
  void _testAllReduce();
  void _fillAllReduceArray(Integer nb_value,CommInfo& c,Parallel::eReduceType rt);
  template<bool UseMessagePassingMng>
  void _testAllReduceArray(Parallel::eReduceType rt);
  template<bool UseMessagePassingMng>
  void _testAllReduceArray2();
  void _testAllReduceArray();
  template<bool UseMessagePassingMng>
  void _testAllGatherVariable3(Int32 root_rank,bool use_generic);
  void _testAllGatherVariable();
  void _testAllGatherVariable2(Int32 root_rank);
  void _fillAllGather(Integer nb_value,CommInfo& c);
  void _testAllGather();
  void _testAllGather2(Int32 root_rank);
  template<bool UseMessagePassingMng>
  void _testAllGather3(Int32 root_rank,bool use_generic);
  void _fillAllToAllVariable(Integer nb_value,AllToAllVariableCommInfo& ci);
  template<bool UseMessagePassingMng>
  void _testAllToAllVariable2();
  void _testAllToAllVariable();
  void _testSendRecv();
  void _testSendRecvNonBlocking();
  void _testSendRecvNonBlocking2();
  void _testSendRecvNonBlocking(Integer nb_message,Integer message_size,bool use_generic);
  void _testMessageProbe(Int32 rank_to_receive,Int32 nb_message,
                         Int32 message_size,bool use_any_source);
  void _testMessageProbe2();
  void _testMessageLegacyProbe(Int32 rank_to_receive,Int32 nb_message,
                                Int32 message_size,bool use_any_source);
  void _testMessageLegacyProbe2();

  MinMaxSumInfo _computeArrayMinMaxSum(ConstArrayView<DataType> values);
  void _checkMinMaxSumFull(const MinMaxSumInfo& expected,const MinMaxSumInfo& current);
  void _checkMinMaxSumOnlyValues(const MinMaxSumInfo& expected,const MinMaxSumInfo& current);

 private:

  IParallelMng* m_parallel_mng;
  String m_datatype_name;
  bool m_verbose;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
doTests()
{
  info() << "Beginning tests for type '" << m_datatype_name << "'";
  _launchTest("MessageProbe",&ThatClass::_testMessageProbe2);
  _launchTest("MessageLegacyProbe",&ThatClass::_testMessageLegacyProbe2);
  _launchTest("AllReduce",&ThatClass::_testAllReduce);
  _launchTest("AllReduce array",&ThatClass::_testAllReduceArray);
  // Le ComputeMinMaxSum n'est pas encore disponible pour les threads avec les Real*
  bool no_minmaxsum = (Generator::isMultiReal() && m_parallel_mng->isThreadImplementation());
  if (!no_minmaxsum)
    _launchTest("ComputeMinMaxSum",&ThatClass::_testComputeMinMaxSum);
  _launchTest("AllGather",&ThatClass::_testAllGather);
  _launchTest("AllGatherVariable",&ThatClass::_testAllGatherVariable);
  _launchTest("Send && Recv",&ThatClass::_testSendRecv);
  _launchTest("Send && Recv non blocking",&ThatClass::_testSendRecvNonBlocking);
  _launchTest("Send && Recv non blocking (2)",&ThatClass::_testSendRecvNonBlocking2);
  _launchTest("AllToAllVariable",&ThatClass::_testAllToAllVariable);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> typename ParallelMngDataTypeTest<DataType>::MinMaxSumInfo
ParallelMngDataTypeTest<DataType>::
_computeArrayMinMaxSum(ConstArrayView<DataType> values)
{
  Integer nb_value = values.size();
  if (nb_value==0)
    ARCANE_THROW(ArgumentException,"Array is empty");

  DataType min_val = values[0];
  DataType max_val = values[0];
  DataType sum_val = values[0];
  Int32 min_rank = 0;
  Int32 max_rank = 0;

  for( Integer z=1; z<nb_value; ++z ){
    DataType val = values[z];
    if (val<min_val){
      min_rank = z;
      min_val = val;
    }
    if (max_val<val){
      max_rank = z;
      max_val = val;
    }
    // Le cast est nécessaire pour le type 'short' car ce
    // dernier est converti en 'int' lors des calculs arithmétiques
    sum_val = (DataType)(sum_val+val);
  }
  return MinMaxSumInfo(min_val,max_val,sum_val,min_rank,max_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie que deux instances de MinMaxSum ont les mêmes
 * valeurs pour les champs min, max et sum.
 */
template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_checkMinMaxSumOnlyValues(const MinMaxSumInfo& expected,const MinMaxSumInfo& current)
{
  if (current.min_val!=expected.min_val)
    ARCANE_FATAL("Bad min_val expected={0} v={1}",expected.min_val,current.min_val);
  if (current.max_val!=expected.max_val)
    ARCANE_FATAL("Bad max_val expected={0} v={1}",expected.max_val,current.max_val);
  // La comparaison en flottant n'est pas forcément strictement égal si l'ordre des opérations
  // n'est pas le même ne séquentiel et parallèle.
  if (!math::isNearlyEqual(current.sum_val,expected.sum_val))
    ARCANE_FATAL("Bad sum_val expected={0} v={1}",expected.sum_val,current.sum_val);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie que deux instances de MinMaxSum sont égales.
 */
template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_checkMinMaxSumFull(const MinMaxSumInfo& expected,const MinMaxSumInfo& current)
{
  _checkMinMaxSumOnlyValues(expected,current);

  if (current.min_rank!=expected.min_rank)
    ARCANE_FATAL("Bad min_rank expected={0} v={1}",expected.min_rank,current.min_rank);
  if (current.max_rank!=expected.max_rank)
    ARCANE_FATAL("Bad max_rank expected={0} v={1}",expected.max_rank,current.max_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_testAllReduce()
{
  IParallelMng* pm = m_parallel_mng;
  Int32 rank = pm->commRank();
  Int32 nb_rank = pm->commSize();
  UniqueArray<DataType> all_values(nb_rank);
  for( Int32 i=0; i<nb_rank; ++i ){
    all_values[i] = Generator::generateMonoValue(i);
    info() << "Value i=" << " v=" << all_values[i];
  }

  DataType my_val = all_values[rank];
  DataType sum_val = pm->reduce(Parallel::ReduceSum,my_val);
  DataType min_val = pm->reduce(Parallel::ReduceMin,my_val);
  DataType max_val = pm->reduce(Parallel::ReduceMax,my_val);
  info() << "InfoFromReduce sum = " << sum_val << " min=" << min_val
         << " max=" << max_val;
  MinMaxSumInfo expected = _computeArrayMinMaxSum(all_values);
  MinMaxSumInfo current(min_val,max_val,sum_val,0,nb_rank-1);

  _checkMinMaxSumOnlyValues(expected,current);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_fillAllReduceArray(Integer nb_value,CommInfo& c,Parallel::eReduceType rt)
{
  IParallelMng* pm = m_parallel_mng;
  Integer rank = pm->commRank();
  Integer nb_rank = pm->commSize();

  c.send_values.resize(nb_value);
  c.recv_values.resize(nb_value);
  c.ref_values.resize(nb_value);

  for( Integer i=0; i<nb_value; ++i ){
    c.send_values[i] = Generator::generateBiValue(rank,i);
  }

  c.ref_values.fill(Generator::zero());

  for( Integer i=0; i<nb_value; ++i ){
    for( Integer z=0; z<nb_rank; ++z ){
      DataType value = Generator::generateBiValue(z,i);
      if (rt==Parallel::ReduceSum){
        // Le cast est nécessaire pour le type 'short' car ce
        // dernier est converti en 'int' lors des calculs arithmétiques
        c.ref_values[i] = (DataType)(c.ref_values[i] + value);
      }
      else if (rt==Parallel::ReduceMin){
        if (z==0 || value<c.ref_values[i])
          c.ref_values[i] = value;
      }
      else if (rt==Parallel::ReduceMax){
        if (z==0 || c.ref_values[i]<value)
          c.ref_values[i] = value;
      }
      else
        throw NotSupportedException(A_FUNCINFO);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_testAllReduceArray()
{
  info() << "Testing AllReduceArray: use IParallelMng";
  _testAllReduceArray2<false>();
  if constexpr (HasMessagePassingMngImplementation<DataType>::hasImpl()){
    info() << "Testing AllReduceArray: use IMessagePassingMng";
    _testAllReduceArray2<true>();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> template<bool UseMessagePassingMng> void
ParallelMngDataTypeTest<DataType>::
_testAllReduceArray2()
{
  info() << "Testing AllReduceArray (Sum) type=" << m_datatype_name;
  _testAllReduceArray<UseMessagePassingMng>(Parallel::ReduceSum);
  info() << "Testing AllReduceArray (Min) type=" << m_datatype_name;
  _testAllReduceArray<UseMessagePassingMng>(Parallel::ReduceMin);
  info() << "Testing AllReduceArray (Max) type=" << m_datatype_name;
  _testAllReduceArray<UseMessagePassingMng>(Parallel::ReduceMax);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> template<bool UseMessagePassingMng> void
ParallelMngDataTypeTest<DataType>::
_testAllReduceArray(Parallel::eReduceType rt)
{
  const Integer nb_size = 5;
  Integer _sizes[nb_size] = { 1 , 9, 143, 3090, 15953 };

  IntegerConstArrayView sizes(nb_size,_sizes);

  IParallelMng* pm = m_parallel_mng;
  [[maybe_unused]] Arccore::MessagePassing::IMessagePassingMng* mpm = pm->messagePassingMng();

  ValueChecker vc(A_FUNCINFO);

  UniqueArray<CommInfo> c(nb_size,CommInfo());
  for( Integer i=0; i<nb_size; ++i ){
    _fillAllReduceArray(sizes[i],c[i],rt);
  }

  // Teste les collectives bloquantes
  for( Integer i=0; i<nb_size; ++i ){
    info() << "Testing AllReduceArray type=" << m_datatype_name << " size=" << sizes[i];
    // Comme le tableau envoyé est aussi utilisé en réception, on le copie
    // sinon il n'aura plus les bonnes valeurs pour les tests non bloquants.
    UniqueArray<DataType> send_copy(c[i].send_values);
    if constexpr (UseMessagePassingMng)
      mpAllReduce(mpm,rt,send_copy.span());
    else
      pm->reduce(rt,send_copy);
    vc.areEqualArray(send_copy.constView(),c[i].ref_values.constView(),"AllReduceArray");
  }

  // Teste les collectives non bloquantes
  // TODO: Pour l'instant pnbc est nul si les collectives ne sont pas supportées.
  IParallelNonBlockingCollective* pnbc = pm->nonBlockingCollective();
  if (pnbc){
    if (pnbc->hasValidReduceForDerivedType()){
      UniqueArray<Parallel::Request> requests;
      for( Integer i=0; i<nb_size; ++i ){
        info() << "Testing NonBlockingAllReduceArray type=" << m_datatype_name << " size=" << sizes[i];
        if constexpr (UseMessagePassingMng)
          requests.add(mpNonBlockingAllReduce(mpm,rt,c[i].send_values,c[i].recv_values));
        else
          requests.add(pnbc->allReduce(rt,c[i].send_values,c[i].recv_values));
      }

      pm->waitAllRequests(requests);

      for( Integer i=0; i<nb_size; ++i ){
        vc.areEqualArray(c[i].recv_values.constView(),c[i].ref_values.constView(),"AllReduceArray NonBlocking");
      }
    }
    else
      info() << "Current MPI implementation does not support MPI_Iallreduce for derived type";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_testComputeMinMaxSum()
{
  info() << "Testing ComputeMixMaxSum for type " << m_datatype_name;
  IParallelMng* pm = m_parallel_mng;
  Int32 rank = pm->commRank();
  Int32 nb_rank = pm->commSize();

  UniqueArray<DataType> all_rank_values(nb_rank);
  for( Integer z=0; z<nb_rank; ++z ){
    DataType rank_val = Generator::generateMonoValue(z);
    info() << "RANK_VAL rank=" << z << " val=" << rank_val;
    all_rank_values[z] = rank_val;
  }
  DataType my_val = all_rank_values[rank];
  MinMaxSumInfo expected_mms = _computeArrayMinMaxSum(all_rank_values);
  DataType zero(Generator::zero());
  DataType min_val(zero);
  DataType max_val(zero);
  DataType sum_val(zero);
  Int32 min_rank = A_NULL_RANK;
  Int32 max_rank = A_NULL_RANK;
  pm->computeMinMaxSum(my_val,min_val,max_val,sum_val,min_rank,max_rank);
  MinMaxSumInfo current_mms(min_val,max_val,sum_val,min_rank,max_rank);
  _checkMinMaxSumFull(expected_mms,current_mms);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_testSendRecv()
{
  IParallelMng* pm = m_parallel_mng;
  Int32 rank = pm->commRank();
  Int32 size = pm->commSize();

  UniqueArray<DataType> buf;
  buf.resize(5);
  if (rank==0){
    for( Integer i=1; i<size; ++i ){
      pm->recv(buf,i);
      for( Integer z=0, n = buf.size(); z<n; ++z ){
        DataType expected = Generator::generateBiValue(i,z);
        if (buf[z]!=expected)
          ARCANE_FATAL("Bad value expected={0} v={1} rank={2} (z={3})",
                       expected,buf[z],i,z);
      }
    }
  }
  else{
    for( Integer z=0, n = buf.size(); z<n; ++z )
      buf[z] = Generator::generateBiValue(rank,z);
    pm->send(buf,0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_testSendRecvNonBlocking()
{
  _testSendRecvNonBlocking(1,12,false);
  _testSendRecvNonBlocking(1,12,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_testSendRecvNonBlocking2()
{
  _testSendRecvNonBlocking(5,25,false);
  _testSendRecvNonBlocking(5,25,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_testMessageProbe2()
{
  // TODO Ajouter un test avec MPI_ANY_TAG
  IParallelMng* pm = m_parallel_mng;
  Int32 comm_size = pm->commSize();
  bool use_any_source = false;
  for( Int32 i=0; i<comm_size; ++i )
    _testMessageProbe(i,5,4082,use_any_source);
  _testMessageProbe(0,3,125,use_any_source);
  _testMessageProbe(1,7,12895,use_any_source);

  use_any_source = true;
  for( Int32 i=0; i<comm_size; ++i )
    _testMessageProbe(i,4,9023,use_any_source);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_testMessageLegacyProbe2()
{
  // TODO Ajouter un test avec MPI_ANY_TAG
  IParallelMng* pm = m_parallel_mng;
  Int32 comm_size = pm->commSize();
  bool use_any_source = false;
  for( Int32 i=0; i<comm_size; ++i )
    _testMessageLegacyProbe(i,5,4082,use_any_source);
  _testMessageLegacyProbe(0,3,125,use_any_source);
  _testMessageLegacyProbe(1,7,12895,use_any_source);

  use_any_source = true;
  for( Int32 i=0; i<comm_size; ++i )
    _testMessageLegacyProbe(i,4,9023,use_any_source);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_testSendRecvNonBlocking(Integer nb_message,Integer message_size,bool use_generic)
{
  IParallelMng* pm = m_parallel_mng;
  Int32 rank = pm->commRank();
  Int32 comm_size = pm->commSize();
  ITraceMng* tm = pm->traceMng();
  if (rank==0){
    UniqueArray<DataType> all_mem_bufs(comm_size*nb_message*message_size);
    Array3View<DataType> all_bufs(all_mem_bufs.data(),comm_size,nb_message,message_size);
    UniqueArray<Parallel::Request> requests;

    for( Integer orig=1; orig<comm_size; ++orig ){
      for( Integer z=0; z<nb_message; ++z ){
        if (use_generic){
          requests.add(pm->receive(all_bufs[orig][z],{MessageRank(orig), Parallel::NonBlocking}));
        }
        else
          requests.add(pm->recv(all_bufs[orig][z],orig,false));
      }
    }

    Arccore::MessagePassing::mpWaitAll(pm->messagePassingMng(),requests);

    for( Integer orig=1; orig<comm_size; ++orig ){
      for( Integer z=0; z<nb_message; ++z ){
        if (m_verbose)
          for( Integer i=0; i<message_size; ++i ){
            tm->info() << "RECV orig=" << orig << " msg=" << z << " i=" << i << " v=" << all_bufs[orig][z][i];
          }
        for( Integer i=0; i<message_size; ++i ){
          DataType expected = Generator::generateBiValue(orig*(nb_message+1),z + i);
          DataType current  = all_bufs[orig][z][i];
          if (current!=expected)
          	ARCANE_FATAL("Bad value expected={0} v={1} rank={2},{3},{4}",
                         expected,current,orig,z,i);
        }
      }
    }
  }
  else{
    UniqueArray2<DataType> all_bufs(nb_message,message_size);
    Ref<Parallel::IRequestList> requests = pm->createRequestListRef();

    for( Integer z=0; z<nb_message; ++z ){
      for( Integer i=0; i<message_size; ++i ){
        all_bufs[z][i] = Generator::generateBiValue(rank*(nb_message+1),z + i);
      }
      if (m_verbose)
        for( Integer i=0; i<message_size; ++i ){
          tm->info() << "SEND orig=" << rank << " msg=" << z << " i=" << i << " v=" << all_bufs[z][i];
        }
      if (use_generic)
        requests->add(pm->send(all_bufs[z],PointToPointMessageInfo(MessageRank(0)).setBlocking(false)));
      else
        requests->add(pm->send(all_bufs[z],0,false));
    }

    requests->wait(Parallel::WaitAll);

  }

  pm->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_testMessageProbe(Int32 rank_to_receive,Integer nb_message,
                  Integer message_size,bool use_any_source)
{
  using Parallel::PointToPointMessageInfo;

  info() << "Test MessageProbe: rank_to_receive=" << rank_to_receive
         << " nb_message=" << nb_message << " size=" << message_size
         << " use_any_source=" << use_any_source;

  IParallelMng* pm = m_parallel_mng;
  Int32 rank = pm->commRank();
  Int32 comm_size = pm->commSize();
  ITraceMng* tm = pm->traceMng();

  if (rank==rank_to_receive){
    // Je dois recevoir \a nb_message de chaque PE (sauf moi)
    UniqueArray<PointToPointMessageInfo> all_msg_info;

    // Nombre de messages déjà recu. Comme les valeurs des éléments du message
    // dépend du nombre de message envoyé par chaque PE, il faut le conserver
    UniqueArray<Int32> ranks_message_number(comm_size,0);

    //UniqueArray<DataType> all_mem_bufs(comm_size*nb_message*message_size);
    //Array3View<DataType> all_bufs(all_mem_bufs.data(),comm_size,nb_message,message_size);
    UniqueArray<Parallel::Request> requests;

    for( Integer orig=0; orig<comm_size; ++orig ){
      if (orig!=rank_to_receive)
        for( Integer z=0; z<nb_message; ++z ){
          MessageRank source_rank( (use_any_source) ? MessageRank::anySourceRank() : MessageRank(orig) );
          all_msg_info.add({source_rank, Parallel::NonBlocking});
        }
    }
    Integer iteration = 0;
    while (!all_msg_info.empty()){
      UniqueArray<PointToPointMessageInfo> new_messages;
      for( const auto& p2p_msg : all_msg_info ){
        MessageId msg = pm->probe(p2p_msg);
        // Limite le nombre d'affichages
        bool do_print = (iteration<50 || (iteration%100)==0);
        if (do_print)
          info() << "I=" << iteration << " MSG=" << p2p_msg << " MSG_ID?=" << msg.isValid();
        if (!msg.isValid()){
          new_messages.add(p2p_msg);
          continue;
        }
        // Effectue un `receive` pour le message sondé.
        // TODO: tester avec des receives non bloquants
        MessageId::SourceInfo si = msg.sourceInfo();
        Int32 orig_rank = si.rank().value();
        if (do_print)
          info() << "I=" << iteration << " VALID_MSG: "
                 << " rank=" << orig_rank << " tag=" << si.tag() << " size=" << si.size();
        UniqueArray<DataType> recv_buf(si.size() / sizeof(DataType));
        // Poste une réception et vérifie les valeurs
        PointToPointMessageInfo msg_info(msg);
        pm->receive(recv_buf,msg_info);
        Int32 msg_number = ranks_message_number[orig_rank];
        ++ranks_message_number[orig_rank];
        for( Integer i=0, n=recv_buf.size(); i<n; ++i ){
          DataType expected = Generator::generateBiValue(orig_rank*(nb_message+1),msg_number + i);
          DataType current  = recv_buf[i];
          if (current!=expected)
            ARCANE_FATAL("Bad value expected={0} v={1} orig_rank={2}, message_number={3} i={4}",
                         expected,current,orig_rank,msg_number,i);
        }
      }
      ++iteration;
      // Fait une petit pause de 1ms pour éviter une boucle trop rapide.
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      if (iteration>25000)
        ARCANE_FATAL("Too many iteration. probably a deadlock");
      all_msg_info = new_messages;
    }
    info() << "END_PROBING";
  }
  else{
    // La taille du message dépend de mon rang. Cela permet de vérifier que
    // IParallelMng::probe() permet bien de récupérer cela.
    message_size += (((message_size / 10) + 1) * pm->commRank()) + 100;
    UniqueArray2<DataType> all_bufs(nb_message,message_size);
    UniqueArray<Parallel::Request> requests;
    PointToPointMessageInfo send_info(MessageRank(rank_to_receive),Parallel::NonBlocking);
    for( Integer z=0; z<nb_message; ++z ){
      for( Integer i=0; i<message_size; ++i ){
        all_bufs[z][i] = Generator::generateBiValue(rank*(nb_message+1),z + i);
      }
      if (m_verbose)
        tm->info() << "SEND orig=" << rank << " msg=" << z << " v=" << all_bufs[z];
      requests.add(pm->send(all_bufs[z],send_info));
    }

    pm->waitAllRequests(requests);
  }

  pm->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_testMessageLegacyProbe(Int32 rank_to_receive,Int32 nb_message,
                        Int32 message_size,bool use_any_source)
{
  using Parallel::PointToPointMessageInfo;
  using Parallel::MessageSourceInfo;

  info() << "Test MessageLegacyProbe: rank_to_receive=" << rank_to_receive
         << " nb_message=" << nb_message << " size=" << message_size
         << " use_any_source=" << use_any_source;

  IParallelMng* pm = m_parallel_mng;
  Int32 rank = pm->commRank();
  Int32 comm_size = pm->commSize();
  ITraceMng* tm = pm->traceMng();

  if (rank==rank_to_receive){
    // Je dois recevoir \a nb_message de chaque PE (sauf moi)
    UniqueArray<PointToPointMessageInfo> all_msg_info;

    // Nombre de messages déjà recu. Comme les valeurs des éléments du message
    // dépend du nombre de message envoyé par chaque PE, il faut le conserver
    UniqueArray<Int32> ranks_message_number(comm_size,0);

    UniqueArray<Parallel::Request> requests;

    for( Integer orig=0; orig<comm_size; ++orig ){
      if (orig!=rank_to_receive)
        for( Integer z=0; z<nb_message; ++z ){
          MessageRank source_rank( (use_any_source) ? MessageRank::anySourceRank() : MessageRank(orig) );
          all_msg_info.add({source_rank, Parallel::NonBlocking});
        }
    }
    Integer iteration = 0;
    while (!all_msg_info.empty()){
      UniqueArray<PointToPointMessageInfo> new_messages;
      for( const auto& p2p_msg : all_msg_info ){
        MessageSourceInfo msg = pm->legacyProbe(p2p_msg);
        // Limite le nombre d'affichages
        bool do_print = (iteration<50 || (iteration%100)==0);
        if (do_print)
          info() << "I=" << iteration << " MSG=" << p2p_msg << " MSG_ID?=" << msg.isValid();
        if (!msg.isValid()){
          //new_messages.add(PointToPointMessageInfo(msg.rank(),msg.tag(),Parallel::NonBlocking));
          new_messages.add(p2p_msg);
          continue;
        }
        // Effectue un `receive` pour le message sondé.
        // TODO: tester avec des receives non bloquants
        MessageSourceInfo si = msg;
        Int32 orig_rank = si.rank().value();
        if (do_print)
          info() << "I=" << iteration << " VALID_MSG: "
                 << " rank=" << orig_rank << " tag=" << si.tag() << " size=" << si.size();
        UniqueArray<DataType> recv_buf(si.size() / sizeof(DataType));
        // Poste une réception et vérifie les valeurs
        PointToPointMessageInfo msg_info(msg.rank(),msg.tag());
        pm->receive(recv_buf,msg_info);
        Int32 msg_number = ranks_message_number[orig_rank];
        ++ranks_message_number[orig_rank];
        for( Integer i=0, n=recv_buf.size(); i<n; ++i ){
          DataType expected = Generator::generateBiValue(orig_rank*(nb_message+1),msg_number + i);
          DataType current  = recv_buf[i];
          if (current!=expected)
            ARCANE_FATAL("Bad value expected={0} v={1} orig_rank={2}, message_number={3} i={4}",
                         expected,current,orig_rank,msg_number,i);
        }
      }
      ++iteration;
      // Fait une petit pause de 1ms pour éviter une boucle trop rapide.
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      if (iteration>25000)
        ARCANE_FATAL("Too many iteration. probably a deadlock");
      all_msg_info = new_messages;
    }
    info() << "END_PROBING";
  }
  else{
    // La taille du message dépend de mon rang. Cela permet de vérifier que
    // IParallelMng::probe() permet bien de récupérer cela.
    message_size += (((message_size / 10) + 1) * pm->commRank()) + 100;
    UniqueArray2<DataType> all_bufs(nb_message,message_size);
    UniqueArray<Parallel::Request> requests;
    PointToPointMessageInfo send_info(MessageRank(rank_to_receive),Parallel::NonBlocking);
    for( Integer z=0; z<nb_message; ++z ){
      for( Integer i=0; i<message_size; ++i ){
        all_bufs[z][i] = Generator::generateBiValue(rank*(nb_message+1),z + i);
      }
      if (m_verbose)
        tm->info() << "SEND orig=" << rank << " msg=" << z << " v=" << all_bufs[z];
      requests.add(pm->send(all_bufs[z],send_info));
    }

    pm->waitAllRequests(requests);
  }

  pm->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_testAllGatherVariable()
{
  _testAllGatherVariable2(-1);
  _testAllGatherVariable2(0);
  Int32 nb_rank = m_parallel_mng->commSize();
  if (nb_rank > 2)
    _testAllGatherVariable2(2);
  if (nb_rank > 6)
    _testAllGatherVariable2(6);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_testAllGatherVariable2(Int32 root_rank)
{
  IParallelMng* pm = m_parallel_mng;
  info() << "Testing AllGatherVariable: use IParallelMng root_rank=" << root_rank;
  _testAllGatherVariable3<false>(root_rank,false);
  if constexpr (HasMessagePassingMngImplementation<DataType>::hasImpl()){
    info() << "Testing AllGatherVariable: use IMessagePassingMng root_rank=" << root_rank;
    _testAllGatherVariable3<true>(root_rank,false);
    if (pm->isParallel() && !pm->isThreadImplementation() && !pm->isHybridImplementation())
      _testAllGatherVariable3<true>(root_rank,true);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> template<bool UseMessagePassingMng> void
ParallelMngDataTypeTest<DataType>::
_testAllGatherVariable3(Int32 root_rank,[[maybe_unused]] bool use_generic)
{
  // root_rank vaut (-1) si on utilise la version collective (allGather)
  IParallelMng* pm = m_parallel_mng;
  Int32 rank = pm->commRank();
  Int32 nb_rank = pm->commSize();
  [[maybe_unused]] Arccore::MessagePassing::IMessagePassingMng* mpm = pm->messagePassingMng();

  UniqueArray<DataType> send_values;
  UniqueArray<DataType> recv_values;

  Integer nb_value = rank + 1;
  
  send_values.resize(nb_value);

  for( Integer i=0; i<nb_value; ++i ){
    send_values[i] = Generator::generateBiValue(rank,i);
  }

  if (root_rank<0){
    if constexpr (UseMessagePassingMng){
      if (use_generic){
        MP::GatherMessageInfo<DataType> gather_info;
        gather_info.setGatherVariable(send_values,&recv_values);
        mpGather(mpm,gather_info);
      }
      else
        mpAllGatherVariable(mpm,send_values,recv_values);
    }
    else
      pm->allGatherVariable(send_values,recv_values);
  }
  else{
    if constexpr (UseMessagePassingMng){
      if (use_generic){
        MP::GatherMessageInfo<DataType> gather_info{MessageRank(root_rank)};
        gather_info.setGatherVariable(send_values,&recv_values);
        mpGather(mpm,gather_info);
        //mpGatherVariable(mpm,send_values,recv_values,root_rank);
      }
      else
        mpGatherVariable(mpm,send_values,recv_values,root_rank);
    }
    else
      pm->gatherVariable(send_values,recv_values,root_rank);
  }

  if (root_rank<0 || root_rank==rank){
    // Verifie tout est OK si on est concerné
    Int32 index = 0;
    for( Integer z=0; z<nb_rank; ++z ){
      Integer count = z + 1;
      for( Integer i=0; i<count; ++i ){
        DataType expected = Generator::generateBiValue(z,i);
        DataType current = recv_values[i+index];
        if (current!=expected)
          ARCANE_FATAL("Bad compare value v={0} expected={1} orig_rank={2} index={3} my_rank={4}",
                       current,expected,z,i,rank);
        if (m_verbose)
          info() << "RECV rank=" << rank << " V[" << i+index << "] = " << recv_values[i+index] << " = " << expected;
      }
      index += count;
    }
  }
  else{
    // Si je ne suis pas concerné par la collective, le tableau de réception
    // doit être vide.
    if (!recv_values.empty())
      ARCANE_FATAL("Receive buffer is not empty");
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_fillAllGather(Integer nb_value,CommInfo& c)
{
  IParallelMng* pm = m_parallel_mng;
  Integer rank = pm->commRank();
  Integer nb_rank = pm->commSize();

  c.send_values.resize(nb_value);
  c.recv_values.resize(nb_value*nb_rank);

  c.ref_values.resize(nb_value*nb_rank);
  {
    Integer index = 0;
    for( Integer z=0; z<nb_rank; ++z ){
      Integer count = nb_value;
      for( Integer i=0; i<count; ++i ){
        c.ref_values[i+index] = Generator::generateBiValue(z,i);
      }
      index += count;
    }
  }

  for( Integer i=0; i<nb_value; ++i ){
    c.send_values[i] = Generator::generateBiValue(rank,i);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_testAllGather()
{
  _testAllGather2(-1);
  _testAllGather2(0);
  Int32 nb_rank = m_parallel_mng->commSize();
  if (nb_rank > 2)
    _testAllGather2(2);
  if (nb_rank > 6)
    _testAllGather2(6);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_testAllGather2(Int32 root_rank)
{
  IParallelMng* pm = m_parallel_mng;
  info() << "Testing AllGather: use IParallelMng root_rank=" << root_rank;
  _testAllGather3<false>(root_rank,false);
  if constexpr (HasMessagePassingMngImplementation<DataType>::hasImpl()){
     info() << "Testing AllGather: use IMessagePassingMng root_rank=" << root_rank;
     _testAllGather3<true>(root_rank,false);
     if (pm->isParallel() && !pm->isThreadImplementation() && !pm->isHybridImplementation())
      _testAllGather3<true>(root_rank,true);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> template<bool UseMessagePassingMng> void
ParallelMngDataTypeTest<DataType>::
_testAllGather3(Int32 root_rank,bool use_generic)
{
  const Integer nb_size = 3;
  Integer _sizes[nb_size] = { 125, 3053, 12950 };
  IntegerConstArrayView sizes(nb_size,_sizes);

  UniqueArray<CommInfo> c(nb_size,CommInfo());
  for( Integer i=0; i<nb_size; ++i ){
    _fillAllGather(sizes[i],c[i]);
  }

  IParallelMng* pm = m_parallel_mng;
  const Int32 rank = pm->commRank();
  [[maybe_unused]] Arccore::MessagePassing::IMessagePassingMng* mpm = pm->messagePassingMng();
  ValueChecker vc(A_FUNCINFO);

  // Teste les collectives bloquantes
  for( Integer i=0; i<nb_size; ++i ){
    info() << "Testing Blocking AllGather with nb_value=" << c[i].send_values.size();
    if (root_rank<0){
      if constexpr (UseMessagePassingMng){
        if (use_generic){
          MP::GatherMessageInfo<DataType> gather_info;
          gather_info.setGather(c[i].send_values,c[i].recv_values);
          mpGather(mpm,gather_info);
        }
        else
          mpAllGather(mpm,c[i].send_values,c[i].recv_values);
      }
      else
        pm->allGather(c[i].send_values,c[i].recv_values);
    }
    else{
      ArrayView<DataType> recv_values = c[i].recv_values;
      if (root_rank!=rank)
        recv_values = ArrayView<DataType>{};
      if constexpr (UseMessagePassingMng){
        if (use_generic){
          MP::GatherMessageInfo<DataType> gather_info{MessageRank(root_rank)};
          gather_info.setGather(c[i].send_values,recv_values);
          mpGather(mpm,gather_info);
        }
        else
          mpGather(mpm,c[i].send_values,recv_values,root_rank);
      }
      else
        pm->gather(c[i].send_values,recv_values,root_rank);
    }
    if (root_rank<0 || root_rank==rank)
      vc.areEqualArray(c[i].recv_values.constView(),c[i].ref_values.constView(),"AllGather");
  }

  // Teste les collectives non bloquantes
  // TODO: Pour l'instant pnbc est nul si les collectives ne sont pas supportées.
  IParallelNonBlockingCollective* pnbc = pm->nonBlockingCollective();
  if (pnbc){
    UniqueArray<Parallel::Request> requests;
    for( Integer i=0; i<nb_size; ++i ){
      info() << "Testing Non Blocking AllGather with nb_value=" << sizes[i];
      c[i].recv_values.fill(Generator::zero());
      if constexpr (UseMessagePassingMng)
        requests.add(mpNonBlockingAllGather(mpm,c[i].send_values,c[i].recv_values));
      else
        requests.add(pnbc->allGather(c[i].send_values,c[i].recv_values));
    }

    pm->waitAllRequests(requests);

    for( Integer i=0; i<nb_size; ++i ){
      vc.areEqualArray(c[i].recv_values.constView(),c[i].ref_values.constView(),"AllGather");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_fillAllToAllVariable(Integer nb_value,AllToAllVariableCommInfo& ci)
{
  IParallelMng* pm = m_parallel_mng;
  ITraceMng* tm = pm->traceMng();
  Integer rank = pm->commRank();

  Integer nb_rank = pm->commSize();
  ci.send_count.resize(nb_rank);
  ci.send_index.resize(nb_rank);
  ci.recv_count.resize(nb_rank);
  ci.recv_index.resize(nb_rank);

  for( Integer i=0; i<nb_rank; ++i ){
    ci.send_count[i] = (rank + i + 1) * nb_value;
    ci.recv_count[i] = (rank + i + 1) * nb_value;
  }

  Integer ntotal_send = 0;
  Integer ntotal_recv = 0;
  for( Integer i=0; i<nb_rank; ++i ){
    ntotal_send += ci.send_count[i];
    ntotal_recv += ci.recv_count[i];
    tm->info() << "Ntotal_send=" << ntotal_send << " total_recv=" << ntotal_recv
           << " s[i]=" << ci.send_count[i] << " r[i]=" << ci.recv_count[i];
  }

  ci.send_values.resize(ntotal_send);
  ci.recv_values.resize(ntotal_recv);

  ci.send_index[0] = 0;
  ci.recv_index[0] = 0;
  for( Integer i=1; i<nb_rank; ++i ){
    ci.send_index[i] = ci.send_index[i-1] + ci.send_count[i-1];
    ci.recv_index[i] = ci.recv_index[i-1] + ci.recv_count[i-1];
  }

  for( Integer i=0; i<nb_rank; ++i ){
    if (m_verbose)
      tm->info() << " I=" << i
                 << " send_count=" << ci.send_count[i]
                 << " send_index=" << ci.send_index[i]
                 << " recv_count=" << ci.recv_count[i]
                 << " recv_index=" << ci.recv_index[i];
    Integer index = ci.send_index[i];
    for( Integer z=0, zn=ci.send_count[i]; z<zn; ++z )
      ci.send_values[index+z] = Generator::generateTriValue(rank,z,i);
  }

  ci.ref_values.resize(ntotal_recv);

  // Remplit les valeurs de référence.
  for( Integer z=0; z<nb_rank; ++z ){
    Int32 index = ci.recv_index[z];
    Integer count = ci.recv_count[z];
    for( Integer i=0; i<count; ++i ){
      DataType expected = Generator::generateTriValue(rank,z,i);
      ci.ref_values[i+index] = expected;
    }
  }

  ci.recv_values.fill(Generator::zero());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ParallelMngDataTypeTest<DataType>::
_testAllToAllVariable()
{
  info() << "Testing AllToAllVariable: use IParallelMng";
  _testAllToAllVariable2<false>();
  if constexpr (HasMessagePassingMngImplementation<DataType>::hasImpl()){
    info() << "Testing AllToAllVariable: use IMessagePassingMng";
    _testAllToAllVariable2<true>();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> template<bool UseMessagePassingMng> void
ParallelMngDataTypeTest<DataType>::
_testAllToAllVariable2()
{
  const Integer nb_size = 5;
  Integer _sizes[nb_size] = { 1, 37, 129, 489, 12950 };
  IntegerConstArrayView sizes(nb_size,_sizes);

  UniqueArray<AllToAllVariableCommInfo> c(nb_size,AllToAllVariableCommInfo());
  for( Integer i=0; i<nb_size; ++i ){
    _fillAllToAllVariable(sizes[i],c[i]);
  }

  ValueChecker vc(A_FUNCINFO);

  IParallelMng* pm = m_parallel_mng;
  [[maybe_unused]] Arccore::MessagePassing::IMessagePassingMng* mpm = pm->messagePassingMng();

  for( Integer i=0; i<nb_size; ++i ){
    info() << "Testing AllToAllVariable with nb_value=" << sizes[i];
    AllToAllVariableCommInfo& ci = c[i];
    if constexpr (UseMessagePassingMng)
      mpAllToAllVariable(mpm,ci.send_values,ci.send_count,ci.send_index,
                         ci.recv_values,ci.recv_count,ci.recv_index);
    else
      pm->allToAllVariable(ci.send_values,ci.send_count,ci.send_index,
                           ci.recv_values,ci.recv_count,ci.recv_index);
    vc.areEqualArray(ci.recv_values.constView(),ci.ref_values.constView(),"AllToAllVariable");
  }

  IParallelNonBlockingCollective* pnbc = pm->nonBlockingCollective();
  if (pnbc){
    UniqueArray<Parallel::Request> requests;
    for( Integer i=0; i<nb_size; ++i ){
      AllToAllVariableCommInfo& ci = c[i];
      info() << "Testing Non Blocking AllToAllVariable with nb_value=" << sizes[i];
      c[i].recv_values.fill(Generator::zero());
      Request rq;
      if constexpr (UseMessagePassingMng)
        rq = mpNonBlockingAllToAllVariable(mpm,ci.send_values,ci.send_count,
                                           ci.send_index,ci.recv_values,ci.recv_count,ci.recv_index);
      else
        rq = pnbc->allToAllVariable(ci.send_values,ci.send_count,
                                    ci.send_index,ci.recv_values,ci.recv_count,ci.recv_index);
      requests.add(rq);
    }

    pm->waitAllRequests(requests);

    for( Integer i=0; i<nb_size; ++i ){
      vc.areEqualArray(c[i].recv_values.constView(),c[i].ref_values.constView(),"AllToAllVariable");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
_testParallelMngDataType(IParallelMng* pm)
{
  ParallelMngDataTypeTest<Real>(pm,"Real").doTests();
  ParallelMngDataTypeTest<Int16>(pm,"Int16").doTests();
  ParallelMngDataTypeTest<Int32>(pm,"Int32").doTests();
  ParallelMngDataTypeTest<Int64>(pm,"Int64").doTests();
  ParallelMngDataTypeTest<Real3>(pm,"Real3").doTests();
  ParallelMngDataTypeTest<Real3x3>(pm,"Real3x3").doTests();
  ParallelMngDataTypeTest<Real2>(pm,"Real2").doTests();
  ParallelMngDataTypeTest<Real2x2>(pm,"Real2x2").doTests();
  ParallelMngDataTypeTest<HPReal>(pm,"HPReal").doTests();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
