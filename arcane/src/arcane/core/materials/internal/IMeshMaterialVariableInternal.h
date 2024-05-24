// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialVariableInternal.h                             (C) 2000-2024 */
/*                                                                           */
/* API interne Arcane de 'IMeshMaterialVariable'.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_INTERNAL_IMESHMATERIALVARIABLEINTERNAL_H
#define ARCANE_CORE_MATERIALS_INTERNAL_IMESHMATERIALVARIABLEINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemTypes.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
class ComponentItemListBuilder;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arguments des méthodes de copie entre valeurs partielles et globales
 */
class ARCANE_CORE_EXPORT MeshVariableCopyBetweenPartialAndGlobalArgs
{
 public:

  MeshVariableCopyBetweenPartialAndGlobalArgs(Int32 var_index,
                                              SmallSpan<const Int32> local_ids,
                                              SmallSpan<const Int32> indexes_in_multiple,
                                              bool do_copy,
                                              RunQueue* queue)
  : m_var_index(var_index)
  , m_local_ids(local_ids)
  , m_indexes_in_multiple(indexes_in_multiple)
  , m_do_copy_between_partial_and_pure(do_copy)
  , m_queue(queue)
  {}

 public:

  Int32 m_var_index = -1;
  SmallSpan<const Int32> m_local_ids;
  SmallSpan<const Int32> m_indexes_in_multiple;
  bool m_do_copy_between_partial_and_pure = true;
  RunQueue* m_queue = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief API interne Arcane de 'IMeshMaterialVariable'.
 */
class ARCANE_CORE_EXPORT IMeshMaterialVariableInternal
{
 public:

  virtual ~IMeshMaterialVariableInternal() = default;

 public:

  /*!
   *\brief Taille en octet pour conserver une valeur de la variable.
   *
   * Pour une variable scalaire, il s'agit de la taille du type de donnée associé.
   * Pour une variable tableau, il s'agit de la taille du type de donnée 
   * multiplié pour le nombre d'éléments du tableau.
   */
  virtual Int32 dataTypeSize() const = 0;

  /*!
   * \brief Copie les valeurs de la variable dans un buffer.
   *
   * \a queue peut être nul.
   */
  virtual void copyToBuffer(SmallSpan<const MatVarIndex> matvar_indexes,
                            Span<std::byte> bytes, RunQueue* queue) const = 0;

  /*!
   * \brief Copie les valeurs de la variable depuis un buffer.
   *
   * \a queue peut être nul.
   */
  virtual void copyFromBuffer(SmallSpan<const MatVarIndex> matvar_indexes,
                              Span<const std::byte> bytes, RunQueue* queue) = 0;

  //! \internal
  virtual Ref<IData> internalCreateSaveDataRef(Integer nb_value) = 0;

  //! \internal
  virtual void saveData(IMeshComponent* component, IData* data) = 0;

  //! \internal
  virtual void restoreData(IMeshComponent* component, IData* data,
                           Integer data_index, Int32ConstArrayView ids, bool allow_null_id) = 0;

  //! \internal
  virtual void copyGlobalToPartial(const MeshVariableCopyBetweenPartialAndGlobalArgs& args) = 0;

  //! \internal
  virtual void copyPartialToGlobal(const MeshVariableCopyBetweenPartialAndGlobalArgs& args) = 0;

  //! \internal
  virtual void initializeNewItems(const ComponentItemListBuilder& list_builder, RunQueue& queue) = 0;

  //! Liste des 'VariableRef' associées à cette variable.
  virtual ConstArrayView<VariableRef*> variableReferenceList() const =0;

  //!Synchronise les références
  virtual void syncReferences(bool check_resize) = 0;

  //! Redimensionne la valeur partielle associée à l'indexer \a index
  virtual void resizeForIndexer(Int32 index, RunQueue& queue) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
