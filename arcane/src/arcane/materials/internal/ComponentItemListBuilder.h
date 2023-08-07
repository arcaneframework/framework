// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemListBuilder.h                                  (C) 2000-2022 */
/*                                                                           */
/* Classe d'aide à la construction d'une liste de ComponentItem.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_COMPONENTITEMLISTBUILDER_H
#define ARCANE_CORE_MATERIALS_COMPONENTITEMLISTBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/core/materials/MatVarIndex.h"

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
class ARCANE_CORE_EXPORT ComponentItemListBuilder
{
 public:

  ComponentItemListBuilder(MeshMaterialVariableIndexer* var_indexer,
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

  Integer m_component_index;
  Integer m_index_in_partial;

  UniqueArray<MatVarIndex> m_pure_matvar_indexes;

  UniqueArray<MatVarIndex> m_partial_matvar_indexes;
  UniqueArray<Int32> m_partial_local_ids;

  MeshMaterialVariableIndexer* m_indexer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

