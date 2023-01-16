// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataReader.h                                               (C) 2000-2018 */
/*                                                                           */
/* Interface de lecture des données d'une variable.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IDATAREADER_H
#define ARCANE_IDATAREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariable;
class IData;
class VariableMetaData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup IO
 \brief Interface de lecture des données d'une variable.

  \sa IDataWriter
 */
class IDataReader
{
 public:

  //! Libère les ressources
  virtual ~IDataReader() {}

 public:
  
  virtual void beginRead(const VariableCollection& vars) =0;
  virtual void endRead() =0;

 public:

  //! Méta-données
  virtual String metaData() =0;

 public:

  //! Lit les données \a data de la variable \a var
  virtual void read(IVariable* var,IData* data) =0;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

