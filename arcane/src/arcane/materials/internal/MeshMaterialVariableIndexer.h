// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableIndexer.h                               (C) 2000-2024 */
/*                                                                           */
/* Indexer for material variables.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_MESHMATERIALVARIABLEINDEXER_H
#define ARCANE_MATERIALS_INTERNAL_MESHMATERIALVARIABLEINDEXER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/materials/MatVarIndex.h"

#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialInfo;
class IMeshEnvironment;
class ComponentItemListBuilder;
class ComponentItemListBuilderOld;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Indexer for material variables.
 *
 * This class contains the information to manage the multi-value part of a
 * material variable.
 */
class ARCANE_MATERIALS_EXPORT MeshMaterialVariableIndexer
: public TraceAccessor
{
  friend class AllEnvData;
  friend class MaterialModifierOperation;
  friend class MeshEnvironment;
  friend class MeshMaterial;
  friend class MeshComponentData;
  friend class MeshMaterialMng;
  friend class IncrementalComponentModifier;
  template <typename DataType> friend class ItemMaterialVariableScalar;

 public:

  MeshMaterialVariableIndexer(ITraceMng* tm, const String& name);

 public:

  //! Name of the indexer
  const String& name() const { return m_name; }

  /*!
   * Size needed to dimension the multiple values for the variables.
   *
   * This is the maximum of the maximum index plus 1.
   */
  Integer maxIndexInMultipleArray() const { return m_max_index_in_multiple_array + 1; }

  Integer index() const { return m_index; }
  ConstArrayView<MatVarIndex> matvarIndexes() const { return m_matvar_indexes; }
  const CellGroup& cells() const { return m_cells; }
  void checkValid();
  //! True if this indexer is for an environment.
  bool isEnvironment() const { return m_is_environment; }
  void dumpStats() const;

 public:

  // Public methods because they are used on accelerators
  void endUpdateAdd(const ComponentItemListBuilder& builder, RunQueue& queue);
  void endUpdateRemoveV2(ConstituentModifierWorkInfo& work_info, Integer nb_remove, RunQueue& queue);
  void transformCells(ConstituentModifierWorkInfo& args, RunQueue& queue, bool is_from_env);

 private:

  //! Private functions but accessible to 'friend' classes.
  //@{
  void endUpdate(const ComponentItemListBuilderOld& builder);
  Array<MatVarIndex>& matvarIndexesArray() { return m_matvar_indexes; }
  void setCells(const CellGroup& cells) { m_cells = cells; }
  void setIsEnvironment(bool is_environment) { m_is_environment = is_environment; }
  void setIndex(Integer index) { m_index = index; }
  Integer nbItem() const { return m_local_ids.size(); }
  ConstArrayView<Int32> localIds() const { return m_local_ids; }

  void changeLocalIds(Int32ConstArrayView old_to_new_ids);
  void endUpdateRemove(ConstituentModifierWorkInfo& args, Integer nb_remove, RunQueue& queue);
  //@}

 private:
 private:

  //! Index of this instance in the list of indexers.
  Integer m_index = -1;

  //! Max index plus 1 in the multiple values array
  Integer m_max_index_in_multiple_array = -1;

  //! Name of the material or environment
  String m_name;

  //! List of meshes for this indexer
  CellGroup m_cells;

  //! List of indices for material variables.
  UniqueArray<MatVarIndex> m_matvar_indexes;

  /*!
   * \brief List of localId() of entities corresponding to m_matvar_indexes.
   * NOTE: eventually, when the traversal is done in the same order as
   * the elements of the group, this array will correspond to that of the localId()
   * of the group and therefore there will be no need to store it.
   * NOTE: note that this array could be useful in case of modification
   * of the mesh (see MeshEnvironment::_changeIds()).
   */
  UniqueArray<Int32> m_local_ids;

  //! True if the indexer is associated with an environment.
  bool m_is_environment = false;

  //! Number of calls to transformation methods
  Int32 m_nb_transform_called = 0;

  /*!
   * \brief Number of useless calls to transformation methods.
   *
   * A call is useless if the list of modified entities in the output
   * is empty.
   */
  Int32 m_nb_useless_add_transform = 0;
  Int32 m_nb_useless_remove_transform = 0;

  //! Indicates whether a message is displayed during a useless transformation
  bool m_is_print_useless_transform = false;

 private:

  static void _changeLocalIdsV2(MeshMaterialVariableIndexer* var_indexer,
                                Int32ConstArrayView old_to_new_ids);
  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
