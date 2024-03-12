// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicReaderWriter.cc                                        (C) 2000-2024 */
/*                                                                           */
/* Lecture/Ecriture simple.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/BasicReaderWriter.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/ISerializedData.h"
#include "arcane/core/MeshUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String BasicReaderWriterCommon::
_getArcaneDBTag()
{
  return "ArcaneCheckpointRestartDataBase";
}

String BasicReaderWriterCommon::
_getOwnMetatadaFile(const String& path, Int32 rank)
{
  StringBuilder filename = path;
  filename += "/own_metadata_";
  filename += rank;
  filename += ".txt";
  return filename;
}

String BasicReaderWriterCommon::
_getArcaneDBFile(const String& path, Int32 rank)
{
  StringBuilder filename = path;
  filename += "/arcane_db_n";
  filename += rank;
  filename += ".acr";
  return filename;
}

String BasicReaderWriterCommon::
_getBasicVariableFile(Int32 version, const String& path, Int32 rank)
{
  if (version >= 3) {
    return _getArcaneDBFile(path, rank);
  }
  StringBuilder filename = path;
  filename += "/var___MAIN___";
  filename += rank;
  filename += ".txt";
  return filename;
}

String BasicReaderWriterCommon::
_getBasicGroupFile(const String& path, const String& name, Int32 rank)
{
  StringBuilder filename = path;
  filename += "/group_";
  filename += name;
  filename += "_";
  filename += rank;
  filename += ".txt";
  return filename;
}

Ref<IDataCompressor> BasicReaderWriterCommon::
_createDeflater(IApplication* app, const String& name)
{
  ServiceBuilder<IDataCompressor> sf(app);
  Ref<IDataCompressor> bc = sf.createReference(app, name);
  return bc;
}

Ref<IHashAlgorithm> BasicReaderWriterCommon::
_createHashAlgorithm(IApplication* app, const String& name)
{
  ServiceBuilder<IHashAlgorithm> sf(app);
  Ref<IHashAlgorithm> bc = sf.createReference(app, name);
  return bc;
}

void BasicReaderWriterCommon::
_fillUniqueIds(const ItemGroup& group, Array<Int64>& uids)
{
  MeshUtils::fillUniqueIds(group.view(),uids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String BasicReaderWriterCommon::
_getMetaDataFileName(Int32 rank) const
{
  StringBuilder filename = m_path;
  filename += "/metadata";
  filename += "-";
  filename += rank;
  filename += ".txt";
  return filename;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicReaderWriterCommon::
BasicReaderWriterCommon(IApplication* app, IParallelMng* pm,
                        const String& path, eOpenMode open_mode)
: TraceAccessor(pm->traceMng())
, m_application(app)
, m_parallel_mng(pm)
, m_open_mode(open_mode)
, m_path(path)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
