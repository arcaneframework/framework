// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MshMeshGenerationInfo.h                                     (C) 2000-2025 */
/*                                                                           */
/* Information about a mesh derived from the 'msh' format.                   */
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
 * \brief Brief information about a mesh derived from the 'msh' format. 
 */
class ARCANE_CORE_EXPORT MshMeshGenerationInfo
{
 public:

  /*!
   * \brief Brief information about a physical name.
   */
  class MshPhysicalName
  {
   public:

    MshPhysicalName(Int32 dimension, Int64 tag, const String& name)
    : m_dimension(dimension)
    , m_tag(tag)
    , m_name(name)
    {}

    //! Constructs a null physical name.
    MshPhysicalName() = default;

   public:

    //! Indicates if the physical name is not defined.
    bool isNull() const { return m_dimension == (-1); }
    Int32 dimension() const { return m_dimension; }
    Int64 tag() const { return m_tag; }
    const String& name() const { return m_name; }

   private:

    Int32 m_dimension = -1;
    Int64 m_tag = -1;
    String m_name;
  };

  /*!
   * \brief Brief information about the '$PhysicalNames' block.
   *
   * This block is optional in the MSH format. Therefore,
   * there is not necessarily a physical name associated with a tag.
   */
  class MshPhysicalNameList
  {
   public:

    void add(Int32 dimension, Int64 tag, const String& name)
    {
      m_physical_names[dimension].add(MshPhysicalName{ dimension, tag, name });
    }
    /*!
     * \brief Retrieves the physical name associated with tag \a tag
     *
     * This name may be null if tag \a tag is not associated
     * with a physical name or if there is no physical name.
     */
    MshPhysicalName find(Int32 dimension, Int64 tag) const
    {
      for (auto& x : m_physical_names[dimension])
        if (x.tag() == tag)
          return x;
      return {};
    }

   private:

    //! List by dimension of the elements of the $PhysicalNames block
    FixedArray<UniqueArray<MshPhysicalName>, 4> m_physical_names;
  };

  //! Info for 0D entities
  class MshEntitiesNodes
  {
   public:

    MshEntitiesNodes(Int64 tag, Int64 physical_tag)
    : m_tag(tag)
    , m_physical_tag(physical_tag)
    {}

   public:

    Int64 tag() const { return m_tag; }
    Int64 physicalTag() const { return m_physical_tag; }

   private:

    Int64 m_tag = -1;
    Int64 m_physical_tag = -1;
  };

  //! Info for 1D, 2D, and 3D entities
  class MshEntitiesWithNodes
  {
   public:

    MshEntitiesWithNodes(Int32 dim, Int64 tag, Int64 physical_tag)
    : m_dimension(dim)
    , m_tag(tag)
    , m_physical_tag(physical_tag)
    {}

   public:

    Int32 dimension() const { return m_dimension; }
    Int64 tag() const { return m_tag; }
    Int64 physicalTag() const { return m_physical_tag; }

   private:

    Int32 m_dimension = -1;
    Int64 m_tag = -1;
    Int64 m_physical_tag = -1;
  };

  class MshPeriodicOneInfo
  {
   public:

    Int32 m_entity_dim = -1;
    Int32 m_entity_tag = -1;
    Int32 m_entity_tag_master = -1;
    //! List of affine values
    UniqueArray<double> m_affine_values;
    // Number of pairs (slave, master)
    Int32 m_nb_corresponding_node = 0;
    //! List of pairs (slave node uniqueId, master node unique())
    UniqueArray<Int64> m_corresponding_nodes;
  };

  //! Information on periodicity
  class MshPeriodicInfo
  {
   public:

    bool hasValues() const { return !m_periodic_list.empty(); }

   public:

    UniqueArray<MshPeriodicOneInfo> m_periodic_list;
  };

 public:

  explicit MshMeshGenerationInfo(IMesh* mesh);

 public:

  static MshMeshGenerationInfo* getReference(IMesh* mesh, bool create);

 public:

  void findEntities(Int32 dimension, Int64 tag, Array<MshEntitiesWithNodes>& entities)
  {
    entities.clear();
    for (auto& x : entities_with_nodes_list[dimension - 1])
      if (x.tag() == tag)
        entities.add(x);
  }

  MshEntitiesNodes* findNodeEntities(Int64 tag)
  {
    for (auto& x : entities_nodes_list)
      if (x.tag() == tag)
        return &x;
    return nullptr;
  }

  MshPhysicalName findPhysicalName(Int32 dimension, Int64 tag) const
  {
    return physical_name_list.find(dimension, tag);
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
