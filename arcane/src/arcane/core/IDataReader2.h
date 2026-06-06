// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataReader2.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface for reading data of a variable.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IDATAREADER2_H
#define ARCANE_CORE_IDATAREADER2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableMetaData;
class IData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Data reading information.
 */
class ARCANE_CORE_EXPORT DataReaderInfo
{
 public:

  DataReaderInfo() {}

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Data reading information for a variable.
 */
class ARCANE_CORE_EXPORT VariableDataReadInfo
{
 public:

  VariableDataReadInfo(VariableMetaData* varmd, IData* data)
  : m_varmd(varmd)
  , m_data(data)
  {}

 public:

  VariableMetaData* variableMetaData() const { return m_varmd; }
  IData* data() const { return m_data; }

 private:

  VariableMetaData* m_varmd;
  IData* m_data;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup IO
 * \brief Interface for reading data of a variable (Version 2)
 *
 * This interface allows reading data of a variable from
 * a checkpoint file.
 *
 * This interface is generally used by
 * IVariableMng::readCheckpoint(). The order of operations
 * is as follows:
 * \code
 * IDataReader2 reader = ...;
 * DataReaderInfo read_infos = ...
 * reader->fillMetaData(...);
 * reader->beginRead(read_infos);
 * for( const VariableDataReadInfo& i : variables )
 *   reader->read(i);
 * reader->endRead();
 * \endcode
 */
class IDataReader2
{
 public:

  //! Releases resources
  virtual ~IDataReader2() {}

 public:

  //! Fills \a bytes with the metadata content
  virtual void fillMetaData(ByteArray& bytes) = 0;
  //! Notifies the start of data reading
  virtual void beginRead(const DataReaderInfo& infos) = 0;
  //! Reads the data specified by \a infos
  virtual void read(const VariableDataReadInfo& infos) = 0;
  //! Notifies the end of data reading
  virtual void endRead() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
