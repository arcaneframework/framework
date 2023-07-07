// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshComponentData.h                                         (C) 2000-2017 */
/*                                                                           */
/* Données d'un constituant (matériau ou milieu) d'un maillage.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHCOMPONENTDATA_H
#define ARCANE_MATERIALS_MESHCOMPONENTDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/ItemGroup.h"

#include "arcane/materials/MatItem.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshMaterialMng;
class MeshEnvironment;
class MatItemVectorView;
class MeshMaterialVariableIndexer;
class MeshComponentPartData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Données d'un constituant (matériau ou milieu) d'un maillage.
 *
 * Cette classe contient les données communes à MeshMaterial et MeshEnvironnment.
 *
 * Cette classe est interne à Arcane.
 */
class MeshComponentData
: public TraceAccessor
{
 public:

  MeshComponentData(IMeshComponent* component,const String& name,Int32 component_id,
                    bool create_indexer);
  ~MeshComponentData() override;

 public:

  const String& name() const { return m_name; }

  MeshMaterialVariableIndexer* variableIndexer() const
  {
    return m_variable_indexer;
  }

  ConstArrayView<ComponentItemInternal*> itemsInternalView() const
  {
    return m_items_internal;
  }

  ArrayView<ComponentItemInternal*> itemsInternalView()
  {
    return m_items_internal;
  }

  void checkValid();

 public:

  const ItemGroup& items() const
  {
    return m_items;
  }

  Int32 componentId() const
  {
    return m_component_id;
  }

 public:

  void resizeItemsInternal(Integer nb_item);
  void setVariableIndexer(MeshMaterialVariableIndexer* indexer);
  void setItems(const ItemGroup& group);
  void changeLocalIdsForInternalList(Int32ConstArrayView old_to_new_ids);
  void rebuildPartData();
  void buildPartData();
  MeshComponentPartData* partData() const { return m_part_data; }

 private:

  //! Constituant dont on gère les données.
  IMeshComponent* m_component;

  /*!
   * \brief Indice du constituant (dans la liste des constituants de ce type).
   * \sa IMeshMaterialMng.
   */
  Int32 m_component_id;

  //! Nom du constituant
  String m_name;

  //! Liste des entités de ce constituant
  ItemGroup m_items;

  //! Indique si on est propriétaire de l'indexeur (dans ce cas on le détruira avec l'instance)
  bool m_is_indexer_owner;

  //! Infos pour l'indexation des variables partielles.
  MeshMaterialVariableIndexer* m_variable_indexer;

  //! Liste des ComponentItemInternal* pour ce constituant.
  UniqueArray<ComponentItemInternal*> m_items_internal;

  MeshComponentPartData* m_part_data;

 private:

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
