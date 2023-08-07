// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableIndexer.h                               (C) 2000-2023 */
/*                                                                           */
/* Indexer pour les variables materiaux.                                     */
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshMaterialVariableIndexerMng;
class MeshMaterialInfo;
class IMeshEnvironment;
class MatItemInternal;
class ComponentItemListBuilder;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Indexer pour les variables materiaux.
 *
 * Cette classe contient les infos pour gérer la partie multi valeur d'une
 * variable matériau.
 */
class MeshMaterialVariableIndexer
: public TraceAccessor
{
  friend class AllEnvData;
  friend class MaterialModifierOperation;
  friend class MeshEnvironment;
  friend class MeshMaterial;
  friend class MeshComponentData;
  friend class MeshMaterialMng;
  template<typename DataType> friend class ItemMaterialVariableScalar;

 public:

  //! Arguments pour transformCells()
  struct TransformCellsArgs
  {
   public:
    TransformCellsArgs(ConstArrayView<bool> cells_to_transform_,
                       Int32Array& pure_local_ids_,
                       Int32Array& partial_indexes_,
                       bool is_add_operation_,
                       bool is_verbose_)
    : cells_to_transform(cells_to_transform_)
    , pure_local_ids(pure_local_ids_)
    , partial_indexes(partial_indexes_)
    , is_add_operation(is_add_operation_)
    , is_verbose(is_verbose_)
    {
    }

   public:

    ConstArrayView<bool> cells_to_transform;
    Int32Array& pure_local_ids;
    Int32Array& partial_indexes;
    bool is_add_operation;
    bool is_verbose;
  };

 public:

  MeshMaterialVariableIndexer(ITraceMng* tm,const String& name);
  MeshMaterialVariableIndexer(const MeshMaterialVariableIndexer& rhs);

 public:

  //! Nom de l'indexeur
  const String& name() const { return m_name; }

  /*!
   * Taille nécessaire pour dimensionner les valeurs multiples pour les variables.
   *
   * Il s'agit du maximum de l'indice maximal plus 1.
   */
  Integer maxIndexInMultipleArray() const { return m_max_index_in_multiple_array+1; }

  Integer index() const { return m_index; }
  ConstArrayView<MatVarIndex> matvarIndexes() const { return m_matvar_indexes; }
  const CellGroup& cells() const { return m_cells; }
  void checkValid();
  //! Vrai si cet indexeur est celui d'un milieu.
  bool isEnvironment() const { return m_is_environment; }

 private:
  
  //! Fonctions publiques mais réservées aux classes de Arcane.
  //@{
  void endUpdate(const ComponentItemListBuilder& builder);
  Array<MatVarIndex>& matvarIndexesArray() { return m_matvar_indexes; }
  void setCells(const CellGroup& cells) { m_cells = cells; }
  void setIsEnvironment(bool is_environment) { m_is_environment = is_environment; }
  void setIndex(Integer index) { m_index = index; }
  Integer nbItem() const { return m_local_ids.size(); }
  ConstArrayView<Int32> localIds() const { return m_local_ids; }

  void changeLocalIds(Int32ConstArrayView old_to_new_ids);
  void transformCells(Int32ConstArrayView nb_env_per_cell,
                      Int32ConstArrayView nb_mat_per_cell,
                      Int32Array& pure_local_ids,
                      Int32Array& partial_indexes,
                      bool is_add_operation, bool is_env,bool is_verbose);
  void endUpdateAdd(const ComponentItemListBuilder& builder);
  void endUpdateRemove(ConstArrayView<bool> removed_local_ids_filter,Integer nb_remove);
  //@}

 private:

  void transformCellsV2(const TransformCellsArgs& args);

 private:

  //! Index de cette instance dans la liste des indexeurs.
  Integer m_index;

  //! Indice max plus 1 dans le tableau des valeurs multiples
  Integer m_max_index_in_multiple_array;

  //! Nom du matériau ou milieu
  String m_name;

  //! Liste des mailles de cet indexer
  CellGroup m_cells;

  //! Liste des indexes pour les variables matériaux.
  UniqueArray<MatVarIndex> m_matvar_indexes;

  /*!
   * \brief Liste des localId() des entités correspondantes à m_matvar_indexes.
   * NOTE: à terme, lorsque le parcours se fera dans le même ordre que
   * les éléments du groupe, ce tableau correspondra à celui des localId()
   * du groupe et il n'y aura donc pas besoin de le stocker.
   * NOTE: à noter que ce tableau pourrait être utile en cas de modification
   * du maillage (voir MeshEnvironment::_changeIds()).
   */
  UniqueArray<Int32> m_local_ids;

  //! Vrai si l'indexeur est associé à un milieu.
  bool m_is_environment;

 private:

  static void _changeLocalIdsV2(MeshMaterialVariableIndexer* var_indexer,
                                Int32ConstArrayView old_to_new_ids);
  void _transformPureToPartial(Int32ConstArrayView nb_env_per_cell,
                               Int32ConstArrayView nb_mat_per_cell,
                               Int32Array& pure_local_ids,
                               Int32Array& partial_indexes,
                               bool is_env,bool is_verbose);
  void _transformPartialToPure(Int32ConstArrayView nb_env_per_cell,
                               Int32ConstArrayView nb_mat_per_cell,
                               Int32Array& pure_local_ids,
                               Int32Array& partial_indexes,
                               bool is_env,bool is_verbose);

  void _transformPureToPartialV2(const TransformCellsArgs& args);
  void _transformPartialToPureV2(const TransformCellsArgs& args);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

