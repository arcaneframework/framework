// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshCompacter.h                                            (C) 2000-2025 */
/*                                                                           */
/* Handling of mesh family compaction.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHCOMPACTER_H
#define ARCANE_CORE_IMESHCOMPACTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Management of mesh family compaction.
 *
 * Instances of this class are created via the manager
 * IMeshCompactMng. Only one compaction can take place at a time.
 *
 * By compaction, we mean any modification of the local numbering
 * of entities within a family. Therefore, gaps may remain in the numbering
 * after calling a compaction (even if this is not the case
 * in the implementations available in %Arcane).
 *
 * Compaction concerns either all families of a mesh, or
 * a single family. The families() method allows returning the
 * list of compacted families.
 *
 * Even if a family is not directly compacted, it participates in
 * certain compaction operations because it may reference compacted
 * entities.
 *
 * The different operations of a compaction are as follows:
 * 1. beginCompact(): calculation of the new local numbering of entities
 * in the compacted families. After calling this method, it is possible
 * to call findCompactInfos() to obtain the correspondences between new
 * and old local numbers for a family.
 * 2. compactVariablesAndGroups(): updating the groups and variables
 * of the compacted families based on this new numbering.
 * 3. updateInternalReferences(): updating references to entities.
 * This concerns all families, not just the compacted ones.
 * 4. endCompact(): finalizes the family compaction. After calling this
 * method, it is no longer possible to retrieve the correspondence information
 * via findCompactInfos().
 * 5. finalizeCompact(): notification to all families that the compaction
 * is finished. This allows, for example, cleaning up or updating
 * certain information.
 *
 * The doAllActions() method allows performing all these phases at once.
 * This is the recommended method for performing a compaction. The following code
 * shows how to perform a compaction on all families:
 *
 * \code
 *
 * IMeshCompactMng* compact_mng = mesh()->_compactMng();
 * IMeshCompacter* compacter = compact_mng->beginCompact();
 *
 * try{
 *   compacter->doAllActions();
 * }
 * catch(...){
 *   compact_mng->endCompact();
 *   throw;
 * }
 * compact_mng->endCompact();
 *
 * \endcode
 */
class ARCANE_CORE_EXPORT IMeshCompacter
{
 public:

  //! Indicates the different phases of compaction
  enum class ePhase
  {
    Init = 0,
    BeginCompact,
    CompactVariableAndGroups,
    UpdateInternalReferences,
    EndCompact,
    Finalize,
    Ended
  };

 public:

  virtual ~IMeshCompacter() = default; //!< Frees resources

 public:

  //! Executes all compaction actions successively.
  virtual void doAllActions() = 0;

  virtual void beginCompact() = 0;
  virtual void compactVariablesAndGroups() = 0;
  virtual void updateInternalReferences() = 0;
  virtual void endCompact() = 0;
  virtual void finalizeCompact() = 0;

  //! Mesh associated with this compacter.
  virtual IMesh* mesh() const = 0;

  /*!
   * \brief Compaction information for the family \a family.
   *
   * The returned pointer may be null if the specified family is not part of the compacted families.
   */
  virtual const ItemFamilyCompactInfos* findCompactInfos(IItemFamily* family) const = 0;

  //! The exchange phase in which we are located.
  virtual ePhase phase() const = 0;

  /*!
   * \brief Indicates whether entities should be sorted during compaction.
   * \pre phase()==ePhase::Init.
   */
  virtual void setSorted(bool v) = 0;

  //! Indicates whether it wishes to sort the entities in addition to compacting them.
  virtual bool isSorted() const = 0;

  //! Families whose entities are compacted.
  virtual ItemFamilyCollection families() const = 0;

  //! \internal
  virtual void _setCompactVariablesAndGroups(bool v) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
