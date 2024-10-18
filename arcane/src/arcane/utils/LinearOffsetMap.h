// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LinearOffsetMap.h                                           (C) 2000-2024 */
/*                                                                           */
/* Liste d'offset linéaires.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_LINEAROFFSETMAP_H
#define ARCANE_UTILS_LINEAROFFSETMAP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Liste d'offset linéaires.
 *
 * `DataType` doit être `Int32` ou `Int64`.
 *
 * \warning Classe expérimentale. A ne pas utiliser en dehors de Arcane.
 */
template <typename DataType>
class LinearOffsetMap
{
 public:

  static_assert(std::is_same_v<DataType, Int32> || std::is_same_v<DataType, Int64>);

 public:

  //! Ajoute un offset \a offset de taille \a size
  ARCANE_UTILS_EXPORT void add(DataType size, DataType offset);

  /*!
   * \brief Récupère un offset suffisant pour un élément de taille \a size.
   *
   * Retourne une valeur négative si aucun offset n'est disponible. Si un offset
   * est disponible retourne sa valeur. L'offset trouvé est supprimé de la liste
   * et on ajoute un offset pour la taille restante si elle n'est pas nulle:
   * si l'offset trouvé est `offset` et que la taille associée est `offset_size`,
   * apelle `add(offset_size - size, offset + size)`.
   */
  ARCANE_UTILS_EXPORT DataType getAndRemoveOffset(DataType size);

  //! Nombre d'éléments dans la table.
  ARCANE_UTILS_EXPORT Int32 size() const;

 private:

  std::multimap<DataType, DataType> m_offset_map;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
