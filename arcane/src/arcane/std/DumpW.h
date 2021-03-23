// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DumpW.h                                                     (C) 2000-2011 */
/*                                                                           */
/* Wrapper de IDataWriter sous l'ancienne interface IDumpW.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_DUMPW_H
#define ARCANE_STD_DUMPW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/IDataWriter.h"
#include "arcane/ArcaneTypes.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/Array2.h"
#include "arcane/utils/Array2View.h"
#include "arcane/utils/MultiArray2.h"
#include "arcane/utils/MultiArray2View.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariable;
class IData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief wrapper transformant des appels à l'interface IDataWriter en ex IDumpW
 */
class ARCANE_STD_EXPORT DumpW 
  : public IDataWriter
{
public:
  //! Constructeur
  DumpW();

  //! Libère les ressources
  virtual ~DumpW();

public:
  
  //! Notifie le début d'écriture
  void beginWrite(const VariableCollection& vars);

  //! Ecrit les données \a data de la variable \a var
  void write(IVariable* var,IData* data);

public:
  //! Notifie la fin d'écriture
  virtual void endWrite() = 0;

  //! Positionne les infos des méta-données
  virtual void setMetaData(const String& meta_data) = 0;

protected:
  //! Visiteur
  class DataVisitor;

  //! Notifie le début d'écriture
  virtual void beginWrite() = 0;

  //! Ecriture pour la variable \a v du tableau \a a
  virtual void writeVal(IVariable& v,ConstArrayView<Byte> a) =0;
  virtual void writeVal(IVariable& v,ConstArrayView<Real> a) =0;
  virtual void writeVal(IVariable& v,ConstArrayView<Int64> a)  =0;
  virtual void writeVal(IVariable& v,ConstArrayView<Int32> a)  =0;
  virtual void writeVal(IVariable& v,ConstArrayView<Real2> a) =0;
  virtual void writeVal(IVariable& v,ConstArrayView<Real3> a)  =0;
  virtual void writeVal(IVariable& v,ConstArrayView<Real2x2> a) =0;
  virtual void writeVal(IVariable& v,ConstArrayView<Real3x3> a)  =0;
  virtual void writeVal(IVariable& v,ConstArrayView<String> a)  =0;

  virtual void writeVal(IVariable& v,ConstArray2View<Byte> a) =0;
  virtual void writeVal(IVariable& v,ConstArray2View<Real> a)  =0;
  virtual void writeVal(IVariable& v,ConstArray2View<Int64> a)  =0;
  virtual void writeVal(IVariable& v,ConstArray2View<Int32> a)  =0;
  virtual void writeVal(IVariable& v,ConstArray2View<Real2> a) =0;
  virtual void writeVal(IVariable& v,ConstArray2View<Real3> a)  =0;
  virtual void writeVal(IVariable& v,ConstArray2View<Real2x2> a) =0;
  virtual void writeVal(IVariable& v,ConstArray2View<Real3x3> a)  =0;

  virtual void writeVal(IVariable& v,ConstMultiArray2View<Byte> a) =0;
  virtual void writeVal(IVariable& v,ConstMultiArray2View<Real> a)  =0;
  virtual void writeVal(IVariable& v,ConstMultiArray2View<Int64> a)  =0;
  virtual void writeVal(IVariable& v,ConstMultiArray2View<Int32> a)  =0;
  virtual void writeVal(IVariable& v,ConstMultiArray2View<Real2> a) =0;
  virtual void writeVal(IVariable& v,ConstMultiArray2View<Real3> a)  =0;
  virtual void writeVal(IVariable& v,ConstMultiArray2View<Real2x2> a) =0;
  virtual void writeVal(IVariable& v,ConstMultiArray2View<Real3x3> a)  =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_STD_DUMPW_H */
