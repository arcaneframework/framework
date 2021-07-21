// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializedData.h                                            (C) 2000-2021 */
/*                                                                           */
/* Donnée sérialisée.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_SERIALIZEDDATA_H
#define ARCANE_IMPL_SERIALIZEDDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/Ref.h"

#include "arcane/ISerializedData.h"

#include "arcane/datatype/DataTypeTraits.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé des données sérialisées.
 *
 * les tableaux \a dimensions et \a values ne sont pas dupliqués et ne doivent
 * pas être modifiés tant que l'objet sérialisé est utilisé.
 *
 * Le type \a data_type doit être un type parmi \a DT_Byte, \a DT_Int16, \a DT_Int32,
 * \a DT_Int64 ou DT_Real.
 */
extern "C++" ARCANE_IMPL_EXPORT
Ref<ISerializedData>
arcaneCreateSerializedDataRef(eDataType data_type,Int64 memory_size,
                              Integer nb_dim,Int64 nb_element,Int64 nb_base_element,
                              bool is_multi_size,Int64ConstArrayView dimensions);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé des données sérialisées.
 *
 * la donnée sérialisée est vide. Elle ne pourra être utilisée qu'après un
 * appel à ISerializedData::serialize() en mode ISerializer::ModePut.
 */
extern "C++" ARCANE_IMPL_EXPORT
Ref<ISerializedData>
arcaneCreateEmptySerializedDataRef();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

