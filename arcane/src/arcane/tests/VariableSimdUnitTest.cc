// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSimdUnitTest.cc                                     (C) 2000-2023 */
/*                                                                           */
/* Service de test des variables.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/SimdOperation.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/ServiceBuildInfo.h"
#include "arcane/ServiceFactory.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/SimdItem.h"
#include "arcane/VariableView.h"
#include "arcane/Timer.h"

#include "arcane/tests/ArcaneTestGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef __GNUG__
#define ARCANE_GCC_VECTORIZE __attribute__((noinline,optimize(2,"tree-vectorize")))
#else
#define ARCANE_GCC_VECTORIZE
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

//! Classe pour tester l'utilisation des vues avec une indirection supplémentaire
template<typename T>
class IndirectArrayView
{
 public:
  IndirectArrayView() = default;
  IndirectArrayView(const ArrayView<T>& h): m_array_view(h)
  {
    m_array_view_ptr = &m_array_view;
  }
  T& operator[](Int32 i) { return (*m_array_view_ptr)[i]; }
  IndirectArrayView& operator=(const ArrayView<T>& v)
  {
    m_array_view = v;
    m_array_view_ptr = &m_array_view;
    return (*this);
  }
 private:
  ArrayView<T>* m_array_view_ptr = nullptr;
  ArrayView<T> m_array_view;
};

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test de la vectorisation sur les variables.
 */
class VariableSimdUnitTest
: public BasicUnitTest
{
 public:

  VariableSimdUnitTest(const ServiceBuildInfo& cb);
  ~VariableSimdUnitTest();

 public:

  virtual void initializeTest();
  virtual void executeTest();

 private:

  VariableCellReal m_var1;
  VariableCellReal m_var2;
  VariableCellReal m_var3;
  VariableCellReal m_var4;

  // Ces classes gardent une référence sur une vue de la variable associée.
  // Il faut donc les initialiser qu'une fois que les variables sont allouées
  IndirectArrayView<Real> m_var1_indirect_array_view;
  IndirectArrayView<Real> m_var2_indirect_array_view;
  IndirectArrayView<Real> m_var3_indirect_array_view;
  IndirectArrayView<Real> m_var4_indirect_array_view;

 private:

  Timer m_timer;
  CellGroup m_cells;

  void _doSimdItem();
  void _doSimdItemIter();
  void _doSimdItemDirect();
  void _doSimdItemDirectLambda();
  void _doItem();
  void _doItemView();
  void _doItemDirect();
  void _doItemDirectDereference();
  void _doItemPointer() ARCANE_GCC_VECTORIZE;
  void _doItemNoIndirect() ARCANE_GCC_VECTORIZE;
  void _doTest( void (VariableSimdUnitTest::*functor)(), const String& name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(VariableSimdUnitTest,IUnitTest,VariableSimdUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSimdUnitTest::
VariableSimdUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
, m_var1(VariableBuildInfo(sb.mesh(),"Var1"))
, m_var2(VariableBuildInfo(sb.mesh(),"Var2"))
, m_var3(VariableBuildInfo(sb.mesh(),"Var3"))
, m_var4(VariableBuildInfo(sb.mesh(),"Var4"))
, m_timer(sb.subDomain(),"Test",Timer::TimerReal)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSimdUnitTest::
~VariableSimdUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSimdUnitTest::
_doSimdItem()
{
  auto out_v1 = viewOut(m_var1);
  auto in_v2 = viewIn(m_var2);
  auto in_v3 = viewIn(m_var3);
  auto in_v4 = viewIn(m_var4);
  ENUMERATE_SIMD_CELL(icell,m_cells){
    SimdCell vi = *icell;
    out_v1[vi] = in_v2[vi] * in_v3[vi] + in_v4[vi];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSimdUnitTest::
_doSimdItemIter()
{
  auto out_v1 = viewOut(m_var1);
  auto in_v2 = viewIn(m_var2);
  auto in_v3 = viewIn(m_var3);
  auto in_v4 = viewIn(m_var4);
  ENUMERATE_SIMD_CELL(icell,m_cells){
    out_v1[icell] = in_v2[icell] * in_v3[icell] + in_v4[icell];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSimdUnitTest::
_doSimdItemDirect()
{
  auto out_v1 = viewOut(m_var1);
  auto in_v2 = viewIn(m_var2);
  auto in_v3 = viewIn(m_var3);
  auto in_v4 = viewIn(m_var4);
  ENUMERATE_SIMD_CELL(icell,m_cells){
    // ATTENTION pour l'instant il n'y a pas de tests de débordement
    // dans SimdItemDirect et donc si le nombre d'éléments de m_cells
    // n'est pas un multiple de la taille d'un vecteur Simd il peut y
    // avoir plantage.
    SimdItemDirectT<Cell> vi(icell.direct());
    out_v1[vi] = in_v2[vi] * in_v3[vi] + in_v4[vi];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename Lambda>
void applySimdDirectLambda(const CellGroup& cells,Lambda& lambda)
{
  ENUMERATE_SIMD_CELL(icell,cells){
    lambda(icell.direct());
  }
}

void VariableSimdUnitTest::
_doSimdItemDirectLambda()
{
  auto out_v1 = viewOut(m_var1);
  auto in_v2 = viewIn(m_var2);
  auto in_v3 = viewIn(m_var3);
  auto in_v4 = viewIn(m_var4);
  auto f = [=](SimdItemDirectT<Cell> vi){
    out_v1[vi] = in_v2[vi] * in_v3[vi] + in_v4[vi];
  };
  applySimdDirectLambda(m_cells,f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSimdUnitTest::
_doItem()
{
  ENUMERATE_CELL(icell,m_cells){
    m_var1[icell] = m_var2[icell] * m_var3[icell] + m_var4[icell];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSimdUnitTest::
_doItemView()
{
  auto out_v1 = viewOut(m_var1);
  auto in_v2 = viewIn(m_var2);
  auto in_v3 = viewIn(m_var3);
  auto in_v4 = viewIn(m_var4);
  ENUMERATE_CELL(icell,m_cells){
    out_v1[icell] = in_v2[icell] * in_v3[icell] + in_v4[icell];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSimdUnitTest::
_doItemDirect()
{
  Int32ConstArrayView idx = m_cells.view().localIds();
  RealArrayView out_v1 = m_var1.asArray();
  RealConstArrayView in_v2 = m_var2.asArray();
  RealConstArrayView in_v3 = m_var3.asArray();
  RealConstArrayView in_v4 = m_var4.asArray();

  for( Integer i=0, n=idx.size(); i<n; ++i ){
    out_v1[idx[i]] = in_v2[idx[i]] * in_v3[idx[i]] + in_v4[idx[i]];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSimdUnitTest::
_doItemDirectDereference()
{
  Int32ConstArrayView idx = m_cells.view().localIds();
  IndirectArrayView<Real> out_v1 = m_var1_indirect_array_view;
  IndirectArrayView<Real> in_v2 = m_var2_indirect_array_view;
  IndirectArrayView<Real> in_v3 = m_var3_indirect_array_view;
  IndirectArrayView<Real> in_v4 = m_var4_indirect_array_view;

  for( Integer i=0, n=idx.size(); i<n; ++i ){
    out_v1[idx[i]] = in_v2[idx[i]] * in_v3[idx[i]] + in_v4[idx[i]];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSimdUnitTest::
_doItemPointer()
{
  Int32ConstArrayView aidx = m_cells.view().localIds();
  RealConstArrayView ain_v2 = m_var2.asArray();
  RealConstArrayView ain_v3 = m_var3.asArray();
  RealConstArrayView ain_v4 = m_var4.asArray();
  RealArrayView aout_v1 = m_var1.asArray();

  const Int32* idx = aidx.data();
  Real* ARCANE_RESTRICT out_v1 = aout_v1.data();
  const Real* in_v2 = ain_v2.data();
  const Real* in_v3 = ain_v3.data();
  const Real* in_v4 = ain_v4.data();

  for( Integer i=0, n=aidx.size(); i<n; ++i ){
    out_v1[idx[i]] = in_v2[idx[i]] * in_v3[idx[i]] + in_v4[idx[i]];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSimdUnitTest::
_doItemNoIndirect()
{
  Int32ConstArrayView aidx = m_cells.view().localIds();
  RealArrayView aout_v1 = m_var1.asArray();
  RealConstArrayView ain_v2 = m_var2.asArray();
  RealConstArrayView ain_v3 = m_var3.asArray();
  RealConstArrayView ain_v4 = m_var4.asArray();

  Real* out_v1 = aout_v1.data();
  const Real* in_v2 = ain_v2.data();
  const Real* in_v3 = ain_v3.data();
  const Real* in_v4 = ain_v4.data();

  for( Integer i=0, n=aidx.size(); i<n; ++i ){
    out_v1[i] = in_v2[i] * in_v3[i] + in_v4[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSimdUnitTest::
_doTest( void (VariableSimdUnitTest::*functor)(), const String& name)
{
  info(4) << "Begin test name=" << name;
  // Multiplier cette valeur par 40 si on veut faire des tests de performances
  Integer nb_z = 500;
  // Réduit la taille du test en débug pour qu'il ne dure pas trop longtemps
  //if (arcaneIsDebug())
  //nb_z /= 20;

  {
    Timer::Sentry ts(&m_timer);
    for( Integer k=0; k<nb_z; ++k ){
      (this->*functor)();
    }
  }
  info() << "TIME = " << m_timer.lastActivationTime() << " name=" << name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSimdUnitTest::
executeTest()
{
  info() << "Execute test nb_cell=" << m_cells.size();
  info() << "Init values";
  ENUMERATE_CELL(icell,m_cells){
    Cell cell = *icell;
    Real x = (Real)(cell.uniqueId().asInt64());
    m_var2[icell] = x * 2.0;
    m_var3[icell] = x + 1.0;
    m_var4[icell] = x * 3.0 - 5.0;
  }
  
  info() << "Do vectorisation";

  // Vérifie que SimdItemDirect fonctionne bien.
  {
    auto out_v1 = viewOut(m_var1);
    auto in_v2 = viewIn(m_var2);
    auto in_v3 = viewIn(m_var3);
    auto in_v4 = viewIn(m_var4);
    // Vérifie conversion DataType -> SimdType.
    ENUMERATE_SIMD_CELL(icell,m_cells){
      out_v1[icell] = 3.2;
    }
    ENUMERATE_SIMD_CELL(icell,m_cells){
      // ATTENTION pour l'instant il n'y a pas de tests de débordement
      // dans SimdItemDirect et donc si le nombre d'éléments de m_cells
      // n'est pas un multiple de la taille d'un vecteur Simd il peut y
      // avoir plantage.
      //SimdItemDirectT<Cell> vi(icell.direct());
      SimdItemT<Cell> vi(*icell);
      out_v1[vi] = in_v2[vi] * in_v3[vi] + in_v4[vi];
    }
    ENUMERATE_CELL(icell,m_cells){
      Real expected = m_var2[icell] * m_var3[icell] + m_var4[icell];
      if (expected!=m_var1[icell]){
        info() << "ERROR: Bad value e=" << expected << " v=" << m_var1[icell] << " lid=" << icell.itemLocalId();
      }
    }
  }

  _doTest(&VariableSimdUnitTest::_doSimdItem,"SimdItem");
  _doTest(&VariableSimdUnitTest::_doSimdItemIter,"SimdItemIter");
  _doTest(&VariableSimdUnitTest::_doSimdItemDirect,"SimdItemDirect");
  _doTest(&VariableSimdUnitTest::_doSimdItemDirectLambda,"SimdItemDirectLambda");
  _doTest(&VariableSimdUnitTest::_doItem,"Item");
  _doTest(&VariableSimdUnitTest::_doItemView,"ItemView");
  _doTest(&VariableSimdUnitTest::_doItemDirect,"ItemDirect");
  _doTest(&VariableSimdUnitTest::_doItemDirectDereference,"ItemDirectDereference");
  _doTest(&VariableSimdUnitTest::_doItemPointer,"ItemPointer");
  _doTest(&VariableSimdUnitTest::_doItemNoIndirect,"ItemNoIndirect");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSimdUnitTest::
initializeTest()
{
  info() << "INITIALIZE TEST";

  info() << "Using vectorisation name=" << SimdInfo::name()
         << " vector_size=" << SimdReal::Length
         << " index_size=" << SimdInfo::Int32IndexSize;

  // Pour tester toutes les configurations, il faut que le nombre de mailles
  // soit un nombre premier. Comme cela on est certain lors des ENUMERATE
  // qu'on ne tombe pas sur un multiple de la taille d'un registre vectoriel.

  Integer n = 10957;
  Int32UniqueArray ids;
  
  {
    Integer index = 0;
    ENUMERATE_CELL(icell,allCells()){
      if (index<n){
        ++index;
        ids.add(icell.itemLocalId());
      }
    }
  }
  if (ids.size()!=n)
    throw FatalErrorException(A_FUNCINFO,"Bad size");
  
  m_cells = mesh()->cellFamily()->createGroup("PrimeCells",ids);

  m_var1_indirect_array_view = m_var1.asArray();
  m_var2_indirect_array_view = m_var2.asArray();
  m_var3_indirect_array_view = m_var3.asArray();
  m_var4_indirect_array_view = m_var4.asArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
