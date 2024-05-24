// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableIndexer.h                               (C) 2000-2024 */
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
 * \brief Indexer pour les variables materiaux.
 *
 * Cette classe contient les infos pour gérer la partie multi valeur d'une
 * variable matériau.
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
  MeshMaterialVariableIndexer(const MeshMaterialVariableIndexer& rhs);

 public:

  //! Nom de l'indexeur
  const String& name() const { return m_name; }

  /*!
   * Taille nécessaire pour dimensionner les valeurs multiples pour les variables.
   *
   * Il s'agit du maximum de l'indice maximal plus 1.
   */
  Integer maxIndexInMultipleArray() const { return m_max_index_in_multiple_array + 1; }

  Integer index() const { return m_index; }
  ConstArrayView<MatVarIndex> matvarIndexes() const { return m_matvar_indexes; }
  const CellGroup& cells() const { return m_cells; }
  void checkValid();
  //! Vrai si cet indexeur est celui d'un milieu.
  bool isEnvironment() const { return m_is_environment; }

 public:

  // Méthodes publiques car utilisées sur accélérateurs
  void endUpdateAdd(const ComponentItemListBuilder& builder, RunQueue& queue);
  void endUpdateRemoveV2(ConstituentModifierWorkInfo& work_info, Integer nb_remove, RunQueue& queue);

 private:

  //! Fonctions publiques mais réservées aux classes de Arcane.
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

  void transformCellsV2(ConstituentModifierWorkInfo& args, RunQueue& queue);

 private:

  //! Index de cette instance dans la liste des indexeurs.
  Integer m_index = -1;

  //! Indice max plus 1 dans le tableau des valeurs multiples
  Integer m_max_index_in_multiple_array = -1;

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
  bool m_is_environment = false;

 private:

  static void _changeLocalIdsV2(MeshMaterialVariableIndexer* var_indexer,
                                Int32ConstArrayView old_to_new_ids);
  void _init();

 public:

  void _switchBetweenPureAndPartial(ConstituentModifierWorkInfo& work_info,
                                    RunQueue& queue,
                                    bool is_pure_to_partial);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

