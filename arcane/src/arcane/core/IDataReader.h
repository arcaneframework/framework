// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataReader.h                                               (C) 2000-2025 */
/*                                                                           */
/* Interface de lecture des données d'une variable.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IDATAREADER_H
#define ARCANE_CORE_IDATAREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup IO
 * \brief Interface de lecture des données d'une variable.
 *
 * \sa IDataWriter
 */
class ARCANE_CORE_EXPORT IDataReader
{
 public:

  //! Libère les ressources
  virtual ~IDataReader() = default;

 public:

  virtual void beginRead(const VariableCollection& vars) = 0;
  virtual void endRead() = 0;

 public:

  //! Méta-données
  virtual String metaData() = 0;

 public:

  //! Lit les données \a data de la variable \a var
  virtual void read(IVariable* var, IData* data) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

