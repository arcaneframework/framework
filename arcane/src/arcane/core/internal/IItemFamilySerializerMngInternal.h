// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilySerializerMngInternal.h                          (C) 2000-2026 */
/*                                                                           */
/* Family serialization/deserialization tool manager.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMFAMILYSERIALIZERMNGINTERNAL_H
#define ARCANE_CORE_IITEMFAMILYSERIALIZERMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IItemFamily;
namespace mesh
{
  namespace PolyhedralTools
  {
    enum class SerializationPolicy{SerializationFamilyOnly,SerializeFamilyAndConnectedFamilies,None};
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Manages the serialization/deserialization of the item families.
 */
class ARCANE_CORE_EXPORT IItemFamilySerializerMngInternal
{
 public:

  virtual ~IItemFamilySerializerMngInternal() = default;

 public:

  /*!
   * \brief Begins the serialization process for item families.
   *
   * \param policy Specifies the serialization policy to use. The policy determines
   * if only the family will be serialized in its serializer of if their connected families would be serialized too.
   * The policy is common to all families.
   */
  virtual void beginSerialization(mesh::PolyhedralTools::SerializationPolicy policy) = 0;

  /*!
   * \brief Finalizes the allocations performed by the serializers registered in the manager.
   *
   * Used for polyhedral meshing where allocations are only performed after
   * all serializations for all families have been completed.
   */
  virtual void finalizeSerialization() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
