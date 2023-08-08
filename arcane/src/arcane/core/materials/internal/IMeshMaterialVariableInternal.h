// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialVariableInternal.h                             (C) 2000-2023 */
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
  virtual Ref<IData> internalCreateSaveDataRef(Integer nb_value) =0;

  //! \internal
  virtual void saveData(IMeshComponent* component,IData* data) =0;

  //! \internal
  virtual void restoreData(IMeshComponent* component,IData* data,Integer data_index,Int32ConstArrayView ids,bool allow_null_id) =0;

  //! \internal
  virtual void copyGlobalToPartial(Int32 var_index,Int32ConstArrayView local_ids,Int32ConstArrayView indexes_in_multiple) =0;

  //! \internal
  virtual void copyPartialToGlobal(Int32 var_index,Int32ConstArrayView local_ids,Int32ConstArrayView indexes_in_multiple) =0;

  //! \internal
  virtual void initializeNewItems(const ComponentItemListBuilder& list_builder) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
