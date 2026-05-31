// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DumpW.h                                                     (C) 2000-2022 */
/*                                                                           */
/* Wrapper of IDataWriter under the old IDumpW interface.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_DUMPW_H
#define ARCANE_STD_DUMPW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IDataWriter.h"
#include "arcane/ArcaneTypes.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/Array2.h"
#include "arcane/utils/Array2View.h"
#include "arcane/utils/MultiArray2.h"
#include "arcane/utils/MultiArray2View.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariable;
class IData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief wrapper transforming calls to the IDataWriter interface into IDumpW
 */
class ARCANE_STD_EXPORT DumpW
: public IDataWriter
{
 public:

  //! Constructor
  DumpW();

  //! Frees resources
  virtual ~DumpW();

 public:

  //! Notifies the start of writing
  void beginWrite(const VariableCollection& vars);

  //! Writes the data \a data of the variable \a var
  void write(IVariable* var, IData* data);

 public:

  //! Notifies the end of writing
  virtual void endWrite() = 0;

  //! Sets metadata information
  virtual void setMetaData(const String& meta_data) = 0;

 protected:

  //! Visitor
  class DataVisitor;

  //! Notifies the start of writing
  virtual void beginWrite() = 0;

  //! Writing for variable \a v of array \a a
  virtual void writeVal(IVariable& v, ConstArrayView<Byte> a) = 0;
  virtual void writeVal(IVariable& v, ConstArrayView<Real> a) = 0;
  virtual void writeVal(IVariable& v, ConstArrayView<Int64> a) = 0;
  virtual void writeVal(IVariable& v, ConstArrayView<Int32> a) = 0;
  virtual void writeVal(IVariable&, ConstArrayView<Int16>)
  {
    throw NotSupportedException(A_FUNCINFO);
  }
  virtual void writeVal(IVariable& v, ConstArrayView<Real2> a) = 0;
  virtual void writeVal(IVariable& v, ConstArrayView<Real3> a) = 0;
  virtual void writeVal(IVariable& v, ConstArrayView<Real2x2> a) = 0;
  virtual void writeVal(IVariable& v, ConstArrayView<Real3x3> a) = 0;
  virtual void writeVal(IVariable& v, ConstArrayView<String> a) = 0;

  virtual void writeVal(IVariable& v, ConstArray2View<Byte> a) = 0;
  virtual void writeVal(IVariable& v, ConstArray2View<Real> a) = 0;
  virtual void writeVal(IVariable& v, ConstArray2View<Int64> a) = 0;
  virtual void writeVal(IVariable& v, ConstArray2View<Int32> a) = 0;
  virtual void writeVal(IVariable&, ConstArray2View<Int16>)
  {
    throw NotSupportedException(A_FUNCINFO);
  }
  virtual void writeVal(IVariable& v, ConstArray2View<Real2> a) = 0;
  virtual void writeVal(IVariable& v, ConstArray2View<Real3> a) = 0;
  virtual void writeVal(IVariable& v, ConstArray2View<Real2x2> a) = 0;
  virtual void writeVal(IVariable& v, ConstArray2View<Real3x3> a) = 0;

  virtual void writeVal(IVariable& v, ConstMultiArray2View<Byte> a) = 0;
  virtual void writeVal(IVariable& v, ConstMultiArray2View<Real> a) = 0;
  virtual void writeVal(IVariable& v, ConstMultiArray2View<Int64> a) = 0;
  virtual void writeVal(IVariable& v, ConstMultiArray2View<Int32> a) = 0;
  virtual void writeVal(IVariable&, ConstMultiArray2View<Int16>)
  {
    throw NotSupportedException(A_FUNCINFO);
  }
  virtual void writeVal(IVariable& v, ConstMultiArray2View<Real2> a) = 0;
  virtual void writeVal(IVariable& v, ConstMultiArray2View<Real3> a) = 0;
  virtual void writeVal(IVariable& v, ConstMultiArray2View<Real2x2> a) = 0;
  virtual void writeVal(IVariable& v, ConstMultiArray2View<Real3x3> a) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
