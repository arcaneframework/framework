// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialModifierOperation.cc                                (C) 2000-2024 */
/*                                                                           */
/* Operation for adding/removing meshes from a material.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/MaterialModifierOperation.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/MemoryUtils.h"
#include "arcane/utils/IMemoryRessourceMng.h"

#include "arcane/core/ItemPrinter.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/materials/IMeshMaterial.h"
#include "arcane/core/materials/internal/IMeshComponentInternal.h"

#include "arcane/materials/internal/MeshMaterialVariableIndexer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MaterialModifierOperation::
MaterialModifierOperation()
: m_ids(MemoryUtils::getDefaultDataAllocator())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MaterialModifierOperation::
MaterialModifierOperation(IMeshMaterial* mat, SmallSpan<const Int32> ids, bool is_add)
: MaterialModifierOperation()
{
  m_mat = mat;
  m_is_add = is_add;
  m_ids.resize(ids.size());
  MemoryUtils::copy<Int32>(m_ids, ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Checks if the meshes \a ids are already in the material \a mat.
 *
 * If \a operation==eOperation::Add, checks that the meshes of \a ids
 * are not already in the material and if \a operation==eOperation::Remove, checks
 * that the meshes of \a ids are in the material.
 *
 * Also checks that an element is present only once in the list \a ids.
 *
 * Returns the number of errors.
 */
Int32 MaterialModifierOperation::
_checkMaterialPresence(MaterialModifierOperation* operation)
{
  IMeshMaterial* mat = operation->material();
  SmallSpan<const Int32> ids = operation->ids();

  const MeshMaterialVariableIndexer* indexer = mat->_internalApi()->variableIndexer();
  IItemFamily* item_family = mat->cells().itemFamily();
  ItemInfoListView items_internal(item_family);
  Integer max_local_id = item_family->maxLocalId();
  UniqueArray<bool> presence_flags(max_local_id, false);
  SmallSpan<const Int32> mat_local_ids = indexer->localIds();
  Integer nb_error = 0;
  String name = mat->name();
  ITraceMng* tm = mat->traceMng();

  for (Int32 lid : ids) {
    if (presence_flags[lid]) {
      tm->info() << "ERROR: item " << ItemPrinter(items_internal[lid])
                 << " is present several times in add/remove list for material mat=" << name;
      ++nb_error;
    }
    presence_flags[lid] = true;
  }

  if (operation->isAdd()) {
    for (Int32 lid : mat_local_ids) {
      if (presence_flags[lid]) {
        tm->info() << "ERROR: item " << ItemPrinter(items_internal[lid])
                   << " is already in material mat=" << name;
        ++nb_error;
      }
    }
  }
  else {
    for (Int32 lid : mat_local_ids) {
      presence_flags[lid] = false;
    }

    for (Int32 lid : ids) {
      if (presence_flags[lid]) {
        tm->info() << "ERROR: item " << ItemPrinter(items_internal[lid])
                   << " is not in material mat=" << name;
        ++nb_error;
      }
    }
  }

  return nb_error;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Filters the array of meshes \a ids so that it is valid.
 *
 * This method allows filtering the values of \a ids so that
 * only valid values remain so that they can be added
 * (if \a do_add is true) or removed (if \a do_add is false) from the material
 * \a mat.
 *
 * The valid values are stored in \a valid_ids.
 */
void MaterialModifierOperation::
_filterValidIds(MaterialModifierOperation* operation, Int32Array& valid_ids)
{
  IMeshMaterial* mat = operation->material();
  const bool do_add = operation->isAdd();
  SmallSpan<const Int32> ids = operation->ids();
  const MeshMaterialVariableIndexer* indexer = mat->_internalApi()->variableIndexer();
  const IItemFamily* item_family = mat->cells().itemFamily();
  Integer max_local_id = item_family->maxLocalId();
  UniqueArray<bool> presence_flags(max_local_id, false);
  SmallSpan<const Int32> mat_local_ids = indexer->localIds();
  ITraceMng* tm = mat->traceMng();

  UniqueArray<Int32> unique_occurence_lids;
  unique_occurence_lids.reserve(ids.size());

  for (Int32 lid : ids) {
    if (!presence_flags[lid]) {
      unique_occurence_lids.add(lid);
      presence_flags[lid] = true;
    }
  }

  valid_ids.clear();

  if (do_add) {
    for (Int32 lid : mat_local_ids) {
      if (presence_flags[lid]) {
        ;
      }
      else
        valid_ids.add(lid);
    }
  }
  else {
    for (Int32 lid : mat_local_ids)
      presence_flags[lid] = false;

    for (Int32 lid : unique_occurence_lids) {
      if (presence_flags[lid]) {
        ;
      }
      else
        valid_ids.add(lid);
    }
  }
  tm->info(4) << "FILTERED_IDS n=" << valid_ids.size() << " ids=" << valid_ids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialModifierOperation::
filterIds()
{
  // TODO: change the default to 'false' and test both configurations
  const bool filter_invalid = true;
  Integer nb_error = _checkMaterialPresence(this);
  if (nb_error != 0) {
    if (filter_invalid) {
      UniqueArray<Int32> filtered_ids;
      _filterValidIds(this, filtered_ids);
      m_ids.swap(filtered_ids);
    }
    else
      ARCANE_FATAL("Invalid values for adding items in material name={0} nb_error={1}",
                   m_mat->name(), nb_error);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
