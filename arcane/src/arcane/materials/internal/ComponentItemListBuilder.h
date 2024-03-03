// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemListBuilder.h                                  (C) 2000-2024 */
/*                                                                           */
/* Classe d'aide à la construction d'une liste de ComponentItem.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_COMPONENTITEMLISTBUILDER_H
#define ARCANE_MATERIALS_INTERNAL_COMPONENTITEMLISTBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/core/materials/MatVarIndex.h"

#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe d'aide à la construction d'une liste de ComponentItem
 * pour un MeshMaterialVariableIndexer.
 */
class ARCANE_MATERIALS_EXPORT ComponentItemListBuilder
{
 public:

  ComponentItemListBuilder(MeshMaterialVariableIndexer* var_indexer);

 public:

  void preAllocate(Int32 nb_item)
  {
    m_pure_matvar_indexes.resize(nb_item);
    m_partial_matvar_indexes.resize(nb_item);
    m_partial_local_ids.resize(nb_item);
  }

  void resize(Int32 nb_pure, Int32 nb_partial)
  {
    m_pure_matvar_indexes.resize(nb_pure);
    m_partial_matvar_indexes.resize(nb_partial);
    m_partial_local_ids.resize(nb_partial);
  }

 public:

  SmallSpan<MatVarIndex> pureMatVarIndexes() { return m_pure_matvar_indexes.view(); }
  SmallSpan<MatVarIndex> partialMatVarIndexes() { return m_partial_matvar_indexes.view(); }
  SmallSpan<Int32> partialLocalIds() { return m_partial_local_ids.view(); }

  SmallSpan<const MatVarIndex> pureMatVarIndexes() const { return m_pure_matvar_indexes.view(); }
  SmallSpan<const MatVarIndex> partialMatVarIndexes() const { return m_partial_matvar_indexes.view(); }
  SmallSpan<const Int32> partialLocalIds() const { return m_partial_local_ids.view(); }

  MeshMaterialVariableIndexer* indexer() const { return m_indexer; }

 private:

  UniqueArray<MatVarIndex> m_pure_matvar_indexes;
  UniqueArray<MatVarIndex> m_partial_matvar_indexes;
  UniqueArray<Int32> m_partial_local_ids;

  MeshMaterialVariableIndexer* m_indexer = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe d'aide à la construction d'une liste de ComponentItem
 * pour un MeshMaterialVariableIndexer.
 */
class ARCANE_MATERIALS_EXPORT ComponentItemListBuilderOld
{
 public:

  ComponentItemListBuilderOld(MeshMaterialVariableIndexer* var_indexer,
                           Integer begin_index_in_partial);

 public:

  //! Ajoute l'entité de localId() \a local_id à la liste des entités pure
  void addPureItem(Int32 local_id)
  {
    m_pure_matvar_indexes.add(MatVarIndex(0,local_id));
  }

  //! Ajoute l'entité de localId() \a local_id à la liste des entités partielles
  void addPartialItem(Int32 local_id)
  {
    //TODO: lorsqu'il y aura la suppression incrémentalle, il faudra
    // aller chercher le bon index dans la liste des index libres de l'indexeur.
    m_partial_matvar_indexes.add(MatVarIndex(m_component_index,m_index_in_partial));
    m_partial_local_ids.add(local_id);
    ++m_index_in_partial;
  }

 public:
  
  ConstArrayView<MatVarIndex> pureMatVarIndexes() const { return m_pure_matvar_indexes; }
  ConstArrayView<MatVarIndex> partialMatVarIndexes() const { return m_partial_matvar_indexes; }
  ConstArrayView<Int32> partialLocalIds() const { return m_partial_local_ids; }
  MeshMaterialVariableIndexer* indexer() const { return m_indexer; }

 private:

  Integer m_component_index = -1;
  Integer m_index_in_partial = -1;

  UniqueArray<MatVarIndex> m_pure_matvar_indexes;

  UniqueArray<MatVarIndex> m_partial_matvar_indexes;
  UniqueArray<Int32> m_partial_local_ids;

  MeshMaterialVariableIndexer* m_indexer = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

