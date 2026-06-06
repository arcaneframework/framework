// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataWriter.h                                               (C) 2000-2025 */
/*                                                                           */
/* Interface for writing variable data.                                      */
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
 \brief Interface for writing variable data.

 During a write operation, the order of calls is as follows:
 \code
 * IDataWriter* writer = ...;
 * writer->beginWrite(vars)
 * writer->setMetaData()
 * foreach(var in vars)
 *   writer->write(var,var_data)
 * writer->endWriter()
 \endcode
 \a vars contains the list of variables that will be saved
  \sa IDataReader
 */
class ARCANE_CORE_EXPORT IDataWriter
{
 public:

  //! Frees resources
  virtual ~IDataWriter() = default;

 public:

  virtual void beginWrite(const VariableCollection& vars) = 0;
  virtual void endWrite() = 0;

 public:

  //! Sets the metadata information
  virtual void setMetaData(const String& meta_data) = 0;

 public:

  //! Writes the data \a data of the variable \a var
  virtual void write(IVariable* var, IData* data) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
