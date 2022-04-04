// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DumpW.cc                                                    (C) 2000-2016 */
/*                                                                           */
/* Wrapper de IDataWriter sous l'ancienne interface IDumpW.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/NotSupportedException.h"

#include "arcane/ArcaneTypes.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Array2.h"
#include "arcane/utils/Array2View.h"
#include "arcane/utils/MultiArray2.h"
#include "arcane/utils/MultiArray2View.h"
#include "arcane/IData.h"
#include "arcane/IVariable.h"
#include "arcane/VariableCollection.h"
#include "arcane/std/DumpW.h"
#include "arcane/AbstractDataVisitor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DumpW::DataVisitor 
: public AbstractDataVisitor
{
public:
  DataVisitor(DumpW * dump, IVariable * var) 
    : m_dump(dump)
    , m_var(var) { }

  virtual ~DataVisitor() { }

public:
  void applyVisitor(IScalarDataT<Byte>* data)  { _applyVisitorT(data); }
  void applyVisitor(IScalarDataT<Real>* data)  { _applyVisitorT(data); }
  void applyVisitor(IScalarDataT<Int16>* data)
  {
    ARCANE_UNUSED(data);
    throw NotSupportedException(A_FUNCINFO);
  }
  void applyVisitor(IScalarDataT<Int32>* data)  { _applyVisitorT(data); }
  void applyVisitor(IScalarDataT<Int64>* data)  { _applyVisitorT(data); }
  void applyVisitor(IScalarDataT<Real2>* data)  { _applyVisitorT(data); }
  void applyVisitor(IScalarDataT<Real3>* data)  { _applyVisitorT(data); }
  void applyVisitor(IScalarDataT<Real2x2>* data)  { _applyVisitorT(data); }
  void applyVisitor(IScalarDataT<Real3x3>* data)  { _applyVisitorT(data); }
  void applyVisitor(IScalarDataT<String>* data)  { _applyVisitorT(data); }

  void applyVisitor(IArrayDataT<Byte>* data) { _applyVisitorT(data); }
  void applyVisitor(IArrayDataT<Real>* data) { _applyVisitorT(data); }
  void applyVisitor(IArrayDataT<Int16>* data)
  {
    ARCANE_UNUSED(data);
    throw NotSupportedException(A_FUNCINFO);
  }
  void applyVisitor(IArrayDataT<Int32>* data) { _applyVisitorT(data); }
  void applyVisitor(IArrayDataT<Int64>* data) { _applyVisitorT(data); }
  void applyVisitor(IArrayDataT<Real2>* data) { _applyVisitorT(data); }
  void applyVisitor(IArrayDataT<Real3>* data) { _applyVisitorT(data); }
  void applyVisitor(IArrayDataT<Real2x2>* data) { _applyVisitorT(data); }
  void applyVisitor(IArrayDataT<Real3x3>* data) { _applyVisitorT(data); }
  void applyVisitor(IArrayDataT<String>* data) { _applyVisitorT(data); }

  void applyVisitor(IArray2DataT<Byte>* data) { _applyVisitorT(data); }
  void applyVisitor(IArray2DataT<Real>* data) { _applyVisitorT(data); }
  void applyVisitor(IArray2DataT<Int16>* data)
  {
    ARCANE_UNUSED(data);
    throw NotSupportedException(A_FUNCINFO);
  }
  void applyVisitor(IArray2DataT<Int32>* data) { _applyVisitorT(data); }
  void applyVisitor(IArray2DataT<Int64>* data) { _applyVisitorT(data); }
  void applyVisitor(IArray2DataT<Real2>* data) { _applyVisitorT(data); }
  void applyVisitor(IArray2DataT<Real3>* data) { _applyVisitorT(data); }
  void applyVisitor(IArray2DataT<Real2x2>* data) { _applyVisitorT(data); }
  void applyVisitor(IArray2DataT<Real3x3>* data) { _applyVisitorT(data); }

  void applyVisitor(IMultiArray2DataT<Byte>* data) { _applyVisitorT(data); }
  void applyVisitor(IMultiArray2DataT<Real>* data) { _applyVisitorT(data); }
  void applyVisitor(IMultiArray2DataT<Int16>* data)
  {
    ARCANE_UNUSED(data);
    throw NotSupportedException(A_FUNCINFO);
  }
  void applyVisitor(IMultiArray2DataT<Int32>* data) { _applyVisitorT(data); }
  void applyVisitor(IMultiArray2DataT<Int64>* data) { _applyVisitorT(data); }
  void applyVisitor(IMultiArray2DataT<Real2>* data) { _applyVisitorT(data); }
  void applyVisitor(IMultiArray2DataT<Real3>* data) { _applyVisitorT(data); }
  void applyVisitor(IMultiArray2DataT<Real2x2>* data) { _applyVisitorT(data); }
  void applyVisitor(IMultiArray2DataT<Real3x3>* data) { _applyVisitorT(data); }

protected:
  template<typename T> void _applyVisitorT(IScalarDataT<T>* data);
  template<typename T> void _applyVisitorT(IArrayDataT<T>* data);
  template<typename T> void _applyVisitorT(IArray2DataT<T>* data);
  template<typename T> void _applyVisitorT(IMultiArray2DataT<T>* data);

private:
  DumpW * m_dump;
  IVariable * m_var;
};

/*---------------------------------------------------------------------------*/

template<typename T>
void 
DumpW::DataVisitor::
_applyVisitorT(IScalarDataT<T>* data) 
{ 
  UniqueArray<T> vtmp(1);
  vtmp[0] = data->value();
  m_dump->writeVal(*m_var,vtmp);
}

/*---------------------------------------------------------------------------*/

template<typename T>
void 
DumpW::DataVisitor::
_applyVisitorT(IArrayDataT<T>* data) 
{ 
  ArrayView<T> view = data->view();
  m_dump->writeVal(*m_var,ArrayView<T>(view.size(),view.data()));
}

/*---------------------------------------------------------------------------*/

template<typename T>
void 
DumpW::DataVisitor::
_applyVisitorT(IArray2DataT<T>* data) 
{ 
  ConstArray2View<T> values = data->view();
  m_dump->writeVal(*m_var,values);
}

/*---------------------------------------------------------------------------*/

template<typename T>
void 
DumpW::DataVisitor::
_applyVisitorT(IMultiArray2DataT<T>* data) 
{ 
  ConstMultiArray2View<T> values = data->value();
  m_dump->writeVal(*m_var,values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DumpW::
DumpW()
{
  ;
}

/*---------------------------------------------------------------------------*/

DumpW::
~DumpW()
{
  ;
}

/*---------------------------------------------------------------------------*/

void DumpW::
beginWrite(const VariableCollection& vars) 
{ 
  ARCANE_UNUSED(vars);
  this->beginWrite();
}

/*---------------------------------------------------------------------------*/

void 
DumpW::
write(IVariable* var,IData* data) 
{ 
  DumpW::DataVisitor v(this,var);
  data->visit(&v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
