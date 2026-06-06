// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataReaderWriter.h                                         (C) 2000-2025 */
/*                                                                           */
/* Interface for reading/writing variable data.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IDATAREADERWRITER_H
#define ARCANE_CORE_IDATAREADERWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IDataReader.h"
#include "arcane/core/IDataWriter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup IO
 * \brief Interface for reading/writing variable data.
 *
 * \sa IDataWriter, IDataReader
 */
class ARCANE_CORE_EXPORT IDataReaderWriter
: public IDataReader
, public IDataWriter
{
 public:

  virtual void free() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
