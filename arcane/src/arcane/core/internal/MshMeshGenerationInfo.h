// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MshMeshGenerationInfo.h                                     (C) 2000-2025 */
/*                                                                           */
/* Informations d'un maillage issu du format 'msh'.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_MSHMESHGENERATIONINFO_H
#define ARCANE_CORE_INTERNAL_MSHMESHGENERATIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/FixedArray.h"
#include "arcane/utils/UniqueArray.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Informations d'un maillage issu du format 'msh'. 
 */
class ARCANE_CORE_EXPORT MshMeshGenerationInfo
{
 public:

  /*!
   * \brief Infos sur un nom physique.
   */
  struct MshPhysicalName
  {
    MshPhysicalName(Int32 _dimension, Int64 _tag, const String& _name)
    : dimension(_dimension)
    , tag(_tag)
    , name(_name)
    {}
    MshPhysicalName() = default;
    bool isNull() const { return dimension == (-1); }
    Int32 dimension = -1;
    Int64 tag = -1;
    String name;
  };

  /*!
   * \brief Infos du bloc '$PhysicalNames'.
   */
  struct MshPhysicalNameList
  {
    void add(Int32 dimension, Int64 tag, const String& name)
    {
      m_physical_names[dimension].add(MshPhysicalName{ dimension, tag, name });
    }
    MshPhysicalName find(Int32 dimension, Int64 tag) const
    {
      for (auto& x : m_physical_names[dimension])
        if (x.tag == tag)
          return x;
      return {};
    }

   private:

    //! Liste par dimension des éléments du bloc $PhysicalNames
    FixedArray<UniqueArray<MshPhysicalName>, 4> m_physical_names;
  };

  //! Infos pour les entités 0D
  struct MshEntitiesNodes
  {
    MshEntitiesNodes(Int64 _tag, Int64 _physical_tag)
    : tag(_tag)
    , physical_tag(_physical_tag)
    {}
    Int64 tag;
    Int64 physical_tag;
  };

  //! Infos pour les entités 1D, 2D et 3D
  class MshEntitiesWithNodes
  {
   public:

    MshEntitiesWithNodes(Int32 _dim, Int64 _tag, Int64 _physical_tag)
    : dimension(_dim)
    , tag(_tag)
    , physical_tag(_physical_tag)
    {}
    Int32 dimension;
    Int64 tag;
    Int64 physical_tag;
  };

  class MshPeriodicOneInfo
  {
   public:

    Int32 m_entity_dim = -1;
    Int32 m_entity_tag = -1;
    Int32 m_entity_tag_master = -1;
    //! Liste des valeurs affines
    UniqueArray<double> m_affine_values;
    // Nombre de couples (esclave, maîtres)
    Int32 m_nb_corresponding_node = 0;
    //! Liste de couples (uniqueId noeud esclave, unique() noeud maître)
    UniqueArray<Int64> m_corresponding_nodes;
  };

  //! Informations sur la périodicité
  class MshPeriodicInfo
  {
   public:

    UniqueArray<MshPeriodicOneInfo> m_periodic_list;
  };

 public:

  explicit MshMeshGenerationInfo(IMesh* mesh);

 public:

  void findEntities(Int32 dimension, Int64 tag, Array<MshEntitiesWithNodes>& entities)
  {
    entities.clear();
    for (auto& x : entities_with_nodes_list[dimension - 1])
      if (x.tag == tag)
        entities.add(x);
  }

  MshEntitiesNodes* findNodeEntities(Int64 tag)
  {
    for (auto& x : entities_nodes_list)
      if (x.tag == tag)
        return &x;
    return nullptr;
  }

 public:

  MshPhysicalNameList physical_name_list;
  UniqueArray<MshEntitiesNodes> entities_nodes_list;
  FixedArray<UniqueArray<MshEntitiesWithNodes>, 3> entities_with_nodes_list;
  MshPeriodicInfo m_periodic_info;

 private:

  IMesh* m_mesh = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
