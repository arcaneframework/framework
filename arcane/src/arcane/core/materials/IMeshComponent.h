// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshComponent.h                                            (C) 2000-2025 */
/*                                                                           */
/* Interface of a component (material or environment) of a mesh.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_CORE_IMESHCOMPONENT_H
#define ARCANE_MATERIALS_CORE_IMESHCOMPONENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
class IMeshComponentInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Interface of a component (material or environment) of a mesh.
 */
class ARCANE_CORE_EXPORT IMeshComponent
{
 public:

  virtual ~IMeshComponent() = default;

 public:

  //! Associated manager.
  virtual IMeshMaterialMng* materialMng() = 0;

  //! Associated trace manager.
  virtual ITraceMng* traceMng() = 0;

  //! Component name
  virtual String name() const = 0;

  /*!
   * \brief Group of meshes for this material.
   *
   * \warning This group must not be modified. To change
   * the number of elements of a material, you must go through
   * the materialMng().
   */
  virtual CellGroup cells() const = 0;

  /*!
   * \brief Component identifier.
   *
   * It is also the index (starting from 0) of this component
   * in the list of components of this type.
   * There is a specific list for materials and environments
   * and therefore a component representing a material can have the
   * same ID as a component representing an environment.
   */
  virtual Int32 id() const = 0;

  /*!
   * \brief Mesh of this component for mesh \a c.
   *
   * If the component is not present in the mesh,
   * the null mesh is returned.
   *
   * The cost of this function is proportional to the number of components
   * present in the mesh.
   */
  virtual ComponentCell findComponentCell(AllEnvCell c) const = 0;

  //! View associated with this component
  virtual ComponentItemVectorView view() const = 0;

  //! Checks that the component is valid.
  virtual void checkValid() = 0;

  //! True if the component is a material
  virtual bool isMaterial() const = 0;

  //! True if the component is an environment
  virtual bool isEnvironment() const = 0;

  //! Indicates if the component is defined for space \a space
  virtual bool hasSpace(MatVarSpace space) const = 0;

  //! View on the list of pure entities (associated with the global mesh) of the component
  virtual ComponentPurePartItemVectorView pureItems() const = 0;

  //! View on the list of impure (partial) entities of the component
  virtual ComponentImpurePartItemVectorView impureItems() const = 0;

  //! View on the pure or impure part of the component's entities
  virtual ComponentPartItemVectorView partItems(eMatPart part) const = 0;

  /*!
   * \brief Returns the component in the form of an IMeshMaterial.
   *
   * If isMaterial()==false, returns \a nullptr
   */
  virtual IMeshMaterial* asMaterial() = 0;

  /*!
   * \brief Returns the component in the form of an IMeshMaterial.
   *
   * If isEnvironment()==false, returns \a nullptr
   */
  virtual IMeshEnvironment* asEnvironment() = 0;

  /*!
   * \brief Sets an execution policy for this constituent
   *
   * \warning This method is experimental. Do not use outside of Arcane.
   *
   * The selected execution policy will be used for
   * creation or modification operations of EnvCellVector,
   * MatCellVector or ComponentItemVector.
   *
   * If \a policy equals Accelerator::eExecutionPolicy::None (the default), the policy of the
   * associated IMeshMaterialMng is used. If it equals Accelerator::eExecutionPolicy::Sequential
   * or Accelerator::eExecutionPolicy::Thread, then execution will take place on the host sequentially
   * or multi-threaded. Other values are invalid.
   *
   * \note The change in execution policy applies to any subsequent modification,
   * even for already created instances of ComponentItemVector.
   */
  virtual void setSpecificExecutionPolicy(Accelerator::eExecutionPolicy policy) = 0;

  /*!
   * \brief Specific execution policy.
   *
   * \sa setSpecificExecutionPolicy().
   */
  virtual Accelerator::eExecutionPolicy specificExecutionPolicy() const = 0;

 public:

  //! Internal API
  virtual IMeshComponentInternal* _internalApi() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
