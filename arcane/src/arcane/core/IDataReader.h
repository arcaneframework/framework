// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataReader.h                                               (C) 2000-2025 */
/*                                                                           */
/* Interface for reading variable data.                                      */
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
 * \brief Interface for reading variable data.
 *
 * \sa IDataWriter
 */
class ARCANE_CORE_EXPORT IDataReader
{
 public:

  //! Frees resources
  virtual ~IDataReader() = default;

 public:

  virtual void beginRead(const VariableCollection& vars) = 0;
  virtual void endRead() = 0;

 public:

  //! Metadata
  virtual String metaData() = 0;

 public:

  //! Reads the data \a data of the variable \a var
  virtual void read(IVariable* var, IData* data) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
