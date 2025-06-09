// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataWriter.h                                               (C) 2000-2025 */
/*                                                                           */
/* Interface d'écriture des données d'une variable.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IDATAWRITER_H
#define ARCANE_CORE_IDATAWRITER_H
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
 \brief Interface d'écriture des données d'une variable.

 Lors d'une écriture, l'ordre d'appel est le suivant:
 \code
 * IDataWriter* writer = ...;
 * writer->beginWrite(vars)
 * writer->setMetaData()
 * foreach(var in vars)
 *   writer->write(var,var_data)
 * writer->endWriter()
 \endcode
 \a vars contient la liste des variables qui vont être sauvées
  \sa IDataReader
 */
class ARCANE_CORE_EXPORT IDataWriter
{
 public:

  //! Libère les ressources
  virtual ~IDataWriter() = default;

 public:

  virtual void beginWrite(const VariableCollection& vars) = 0;
  virtual void endWrite() = 0;

 public:

  //! Positionne les infos des méta-données
  virtual void setMetaData(const String& meta_data) = 0;

 public:

  //! Ecrit les données \a data de la variable \a var
  virtual void write(IVariable* var, IData* data) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

